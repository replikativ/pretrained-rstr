(ns pretrained.continuation.paged-decoder
  "Paged autoregressive execution over an already-bound resident decoder.

  The generated layer is replayed in two resident stages around Raster's paged
  K/V append and attention graphs. Q, K, V, and attention output stay in device
  buffers; the host moves only token/position metadata and the selected token."
  (:refer-clojure :exclude [run!])
  (:require [pretrained.attention-state :as attention-state]
            [pretrained.continuation.page-pool :as page-pool]
            [pretrained.continuation.paged-append :as paged-append]
            [pretrained.continuation.paged-attention :as paged-attention]
            [pretrained.decoder-gpu :as decoder-gpu]
            [raster.gpu.core :as gpu])
  (:import [java.io Closeable]
           [java.util UUID]))

(declare close!)

(defrecord PagedDecoder
           [decode-state pool append-runners attention-runners graph-keys state]
  Closeable
  (close [decoder]
    (close! decoder)))

(defn paged-decoder?
  "Return true when `value` is a paged decoder."
  [value]
  (instance? PagedDecoder value))

(defn- checked-positive
  [field value]
  (when-not (and (integer? value) (pos? value))
    (throw (ex-info "Paged decoder capacity must be a positive integer"
                    {:field field :value value})))
  (long value))

(defn- require-open!
  [decoder]
  (when (:closed? @(:state decoder))
    (throw (ex-info "Paged decoder is closed" {}))))

(defn- phase-layout!
  [decode-state]
  (let [layout (:phase-layout decode-state)
        layers (:layers layout)]
    (when-not (and (= (:n-layers (:model decode-state)) (count layers))
                   (every? #(and (seq (:pre %)) (seq (:post %))) layers)
                   (seq (:head layout)) (seq (:tail layout)))
      (throw (ex-info "Decode state has no complete staged phase layout"
                      {:phase-layout layout})))
    layout))

(defn open!
  "Attach paged K/V execution to a resident state returned by `bind-decode!`.

  Options:
  - `:page-size` defaults to 16 tokens.
  - `:physical-pages` defaults to enough pages for one `:maxpos` continuation.
  - `:key-prefix` optionally supplies deterministic session buffer names.

  The decoder owns its runners but not the decode state, Raster session, or page
  pool allocations. Closing it releases runner graphs and descriptor buffers;
  close the decode state's session to release all resident tensors."
  [decode-state & {:keys [page-size physical-pages key-prefix]
                   :or {page-size 16}}]
  (when-not (= :paged (:cache-mode decode-state))
    (throw (ex-info "Paged decoder requires bind-decode! with :cache-mode :paged"
                    {:cache-mode (:cache-mode decode-state)})))
  (let [{:keys [sess model maxpos]} decode-state
        layout (phase-layout! decode-state)
        page-size (checked-positive :page-size page-size)
        pages-per-sequence (long (quot (+ (long maxpos) (dec page-size)) page-size))
        physical-pages (checked-positive :physical-pages
                                         (or physical-pages pages-per-sequence))
        prefix (or key-prefix (str "paged-decoder-" (UUID/randomUUID)))
        pool (page-pool/open-pool!
              sess (attention-state/layout model)
              {:page-size page-size
               :physical-pages physical-pages
               :dtype :half
               :key-prefix (str prefix "-pool")})
        q-elements (* (long (:n-q model)) (long (:head-dim model)))
        kv-elements (* (long (:n-kv model)) (long (:head-dim model)))
        query-view (gpu/buffer-view sess :qr {:shape [q-elements]
                                              :id [prefix :query]})
        key-view (gpu/buffer-view sess :kr {:shape [kv-elements]
                                            :id [prefix :key]})
        value-view (gpu/buffer-view sess :v {:shape [kv-elements]
                                             :id [prefix :value]})
        output-view (gpu/buffer-view sess :at {:shape [q-elements]
                                               :id [prefix :output]})
        graph-keys
        (mapv (fn [layer-index {:keys [pre post]}]
                (let [pre-key [::pre prefix layer-index]
                      post-key [::post prefix layer-index]]
                  (gpu/record-graph! sess pre pre-key)
                  (gpu/record-graph! sess post post-key)
                  {:pre pre-key :post post-key}))
              (range) (:layers layout))
        head-tail-key [::head-tail prefix]
        append-runners (atom [])
        attention-runners (atom [])]
    (gpu/record-graph! sess (into (:head layout) (:tail layout)) head-tail-key)
    (try
      (doseq [layer (range (:n-layers model))]
        (swap! append-runners conj
               (paged-append/open-runner!
                pool {:id [::append prefix layer]
                      :key-prefix (str prefix "-append-" layer)
                      :layer layer
                      :batch-size 1
                      :key-view key-view
                      :value-view value-view}))
        (swap! attention-runners conj
               (paged-attention/open-runner!
                pool {:id [::attention prefix layer]
                      :key-prefix (str prefix "-attention-" layer)
                      :layer layer
                      :batch-size 1
                      :total-query-tokens 1
                      :q-heads (:n-q model)
                      :kv-heads (:n-kv model)
                      :qk-head-dim (:head-dim model)
                      :value-head-dim (:head-dim model)
                      :pages-per-sequence pages-per-sequence
                      :scale (:attn-scale model)
                      :query-dtype :float
                      :output-dtype :float
                      :query-view query-view
                      :output-view output-view})))
      (map->PagedDecoder
       {:decode-state decode-state
        :pool pool
        :append-runners @append-runners
        :attention-runners @attention-runners
        :graph-keys {:layers graph-keys :head-tail head-tail-key}
        :state (atom {:closed? false})})
      (catch Throwable error
        (doseq [runner (concat @attention-runners @append-runners)]
          (.close ^Closeable runner))
        (throw error)))))

(defn allocate-continuation!
  "Allocate an empty resident route for `continuation-id` and return it.

  `:start-position` defaults to zero and permits restored context windows whose
  first cached token has a nonzero absolute position."
  [decoder continuation-id & {:keys [start-position]
                              :or {start-position 0}}]
  (require-open! decoder)
  (page-pool/allocate-route! (:pool decoder) continuation-id 0
                             {:start-position start-position}))

(defn prime-token!
  "Put `token` in the shared resident input row without advancing its route."
  [decoder token]
  (require-open! decoder)
  (decoder-gpu/prime-resident-token! (:decode-state decoder) token)
  decoder)

(defn step!
  "Process the resident input token and append one K/V row atomically.

  `position` must be the route's next absolute position. Every layer writes the
  reserved physical slot and consumes the prospective route before the logical
  token count is published. On failure the reservation is rolled back and its
  partially written slot remains unreachable. Returns the next greedy token id."
  [decoder continuation-id position]
  (require-open! decoder)
  (let [{:keys [sess]} (:decode-state decoder)
        route (or (page-pool/route (:pool decoder) continuation-id)
                  (throw (ex-info "Continuation is not resident"
                                  {:continuation-id continuation-id})))
        expected-position (+ (long (:start-position route))
                             (long (:token-count route)))]
    (when-not (= expected-position (long position))
      (throw (ex-info "Paged decode position does not extend the resident route"
                      {:continuation-id continuation-id
                       :expected expected-position :actual position})))
    (when-not (< (long (:token-count route))
                 (long (:maxpos (:decode-state decoder))))
      (throw (ex-info "Paged decode reached its route capacity"
                      {:continuation-id continuation-id
                       :token-count (:token-count route)
                       :capacity (:maxpos (:decode-state decoder))})))
    (gpu/upload! sess :posbuf (long-array [position]))
    (gpu/upload! sess :clenbuf (long-array [(inc (long position))]))
    (let [batch (paged-append/reserve-batch! (:pool decoder) [continuation-id])]
      (try
        (doseq [[layer-graphs append-runner attention-runner]
                (map vector (get-in decoder [:graph-keys :layers])
                     (:append-runners decoder) (:attention-runners decoder))]
          (gpu/replay! sess (:pre layer-graphs))
          (paged-append/run! append-runner batch)
          (paged-attention/run!
           attention-runner
           {:continuation-ids [continuation-id]
            :append-reservations (paged-append/reservation-entries batch)
            :row-offsets [0 1]
            :positions [position]})
          (gpu/replay! sess (:post layer-graphs)))
        (paged-append/commit-batch! batch)
        (gpu/replay! sess (get-in decoder [:graph-keys :head-tail]))
        (long (aget ^ints (gpu/download sess :tokbuf) 0))
        (catch Throwable error
          (when (= :reserved @(:state batch))
            (paged-append/abort-batch! batch))
          (throw error))))))

(defn decode-token!
  "Upload `token`'s embedding, process it at `position`, and return the next token."
  [decoder continuation-id token position]
  (prime-token! decoder token)
  (step! decoder continuation-id position))

(defn generate!
  "Greedily generate up to `max-new` ids using a paged continuation route.

  The route must be empty. Prompt tokens except the last are first committed to
  K/V; the last token is then processed by the rollout. Generation stops at an
  id in `eos-ids` or at the decode state's maximum position."
  [decoder continuation-id prompt max-new & {:keys [eos-ids]
                                             :or {eos-ids #{}}}]
  (require-open! decoder)
  (when-not (seq prompt)
    (throw (ex-info "Paged generation requires a nonempty prompt" {})))
  (let [route (or (page-pool/route (:pool decoder) continuation-id)
                  (allocate-continuation! decoder continuation-id))]
    (when-not (zero? (:token-count route))
      (throw (ex-info "Paged generation requires an empty continuation route"
                      {:continuation-id continuation-id
                       :token-count (:token-count route)})))
    (let [start (long (:start-position route))
          prefix-count (dec (count prompt))]
      (doseq [offset (range prefix-count)]
        (decode-token! decoder continuation-id (nth prompt offset) (+ start offset)))
      (prime-token! decoder (last prompt))
      (loop [position (+ start prefix-count)
             output []]
        (if (and (< (count output) (long max-new))
                 (< (:token-count (page-pool/route (:pool decoder) continuation-id))
                    (long (:maxpos (:decode-state decoder)))))
          (let [token (step! decoder continuation-id position)
                output (conj output token)]
            (if (contains? eos-ids token)
              output
              (recur (inc position) output)))
          output)))))

(defn close!
  "Release runner graphs and descriptors. Idempotent; does not close the session."
  [decoder]
  (locking decoder
    (when-not (:closed? @(:state decoder))
      (doseq [runner (concat (:attention-runners decoder)
                             (:append-runners decoder))]
        (.close ^Closeable runner))
      (swap! (:state decoder) assoc :closed? true)))
  nil)
