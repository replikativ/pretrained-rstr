(ns pretrained.continuation.paged-attention
  "Raster routed-attention binding for resident continuation page pools.

  A runner owns fixed-capacity route buffers and one verified Raster kernel
  graph. Query and output may either be private buffers or caller-owned Raster
  resident views, allowing adjacent model graphs to compose without host tensor
  transfers. Buffer contents vary per batch; page-pool addresses and graph
  geometry remain stable."
  (:refer-clojure :exclude [run!])
  (:require [pretrained.continuation.page-pool :as page-pool]
            [raster.compiler.ir.attention :as attention]
            [raster.compiler.passes.parallel.attention-route :as attention-route]
            [raster.gpu.core :as gpu])
  (:import [java.io Closeable]
           [java.util UUID]))

(declare close-runner!)

(defrecord PagedAttentionRunner
           [session pool problem handle buffer-keys graph-key state]
  Closeable
  (close [runner]
    (close-runner! runner)))

(defn runner?
  "Return true when `value` is a paged-attention runner."
  [value]
  (instance? PagedAttentionRunner value))

(defn- checked-positive
  [field value]
  (when-not (and (integer? value) (pos? value))
    (throw (ex-info "Paged-attention capacity must be a positive integer"
                    {:field field :value value})))
  (long value))

(defn- checked-io-dtype
  [field value]
  (when-not (contains? #{:half :float} value)
    (throw (ex-info "Paged-attention tensor dtype must be :half or :float"
                    {:field field :value value})))
  value)

(defn- scoped-key
  [prefix role]
  (keyword (str prefix "-" (name role))))

(defn- checked-resident-view
  [role view]
  (when (and view (not (gpu/resident-buffer-view? view)))
    (throw (ex-info "Paged-attention external bindings must be Raster resident views"
                    {:role role :binding view :actual (type view)})))
  view)

(defn- slab
  [pool slab-name layer expected-elements]
  (let [slab-layout (some #(when (= slab-name (:name %)) %)
                          (:slabs (:layout pool)))]
    (when-not slab-layout
      (throw (ex-info "Paged attention requires key and value slabs"
                      {:missing slab-name :layout (:layout pool)})))
    (when-not (and (<= 0 (long layer))
                   (< (long layer) (:count slab-layout)))
      (throw (ex-info "Paged-attention layer is outside its state slab"
                      {:slab slab-name :layer layer :layers (:count slab-layout)})))
    (when-not (= expected-elements (:elements-per-token slab-layout))
      (throw (ex-info "Paged-attention dimensions differ from the page-pool slab"
                      {:slab slab-name :expected expected-elements
                       :actual (:elements-per-token slab-layout)})))
    slab-layout))

(defn reference-plan
  "Build Raster's executable FP16-KV reference attention plan without touching a GPU.

  Required options are `:layer`, `:batch-size`, `:total-query-tokens`,
  `:q-heads`, `:kv-heads`, `:qk-head-dim`, `:value-head-dim`, and
  `:pages-per-sequence`. `:scale` defaults to `1/sqrt(qk-head-dim)`.
  `:query-dtype` and `:output-dtype` accept `:half` (the default) or `:float`.
  Optional `:query-view` and `:output-view` bind caller-owned Raster resident
  views of those dtypes instead of allocating private tensor buffers.

  Returns the semantic problem, verified graph, logical buffer identities and
  concrete allocation specs used by `open-runner!`."
  [pool {:keys [id layer batch-size total-query-tokens q-heads kv-heads
                qk-head-dim value-head-dim pages-per-sequence scale key-prefix
                query-view output-view query-dtype output-dtype]
         :as opts}]
  (when-not (page-pool/page-pool? pool)
    (throw (ex-info "Paged attention requires a DevicePagePool" {:pool pool})))
  (when-not (= :half (:dtype pool))
    (throw (ex-info "Raster reference attention currently requires an FP16 page pool"
                    {:pool-dtype (:dtype pool)})))
  (let [layer (long (or layer 0))
        batch-size (checked-positive :batch-size batch-size)
        total-query-tokens (checked-positive :total-query-tokens total-query-tokens)
        q-heads (checked-positive :q-heads q-heads)
        kv-heads (checked-positive :kv-heads kv-heads)
        qk-head-dim (checked-positive :qk-head-dim qk-head-dim)
        value-head-dim (checked-positive :value-head-dim value-head-dim)
        pages-per-sequence (checked-positive :pages-per-sequence pages-per-sequence)
        query-dtype (checked-io-dtype :query-dtype (or query-dtype :half))
        output-dtype (checked-io-dtype :output-dtype (or output-dtype :half))
        _ (slab pool :key layer (* kv-heads qk-head-dim))
        _ (slab pool :value layer (* kv-heads value-head-dim))
        prefix (or key-prefix (str "paged-attention-" (UUID/randomUUID)))
        ids (into {}
                  (map (fn [role] [role (keyword (str prefix "/" (name role)))]))
                  [:query :query-row-offsets :query-positions
                   :key-pages :value-pages :page-table :lengths
                   :start-positions :output])
        keys (into {}
                   (map (fn [role] [role (scoped-key prefix role)]))
                   [:query :query-row-offsets :query-positions
                    :page-table :lengths :start-positions :output])
        query (attention/packed-query-batch
               {:values (:query ids)
                :row-offsets (:query-row-offsets ids)
                :positions (:query-positions ids)
                :total-tokens total-query-tokens})
        route (attention/dense-paged-route
               {:page-table (:page-table ids)
                :lengths (:lengths ids)
                :start-positions (:start-positions ids)
                :pages-per-sequence pages-per-sequence})
        problem (attention/make
                 {:id (or id [:pretrained-paged-attention prefix])
                  :query query
                  :k-pages (:key-pages ids)
                  :v-pages (:value-pages ids)
                  :route route
                  :output (:output ids)
                  :batch-size batch-size
                  :q-heads q-heads
                  :kv-heads kv-heads
                  :qk-head-dim qk-head-dim
                  :value-head-dim value-head-dim
                  :page-size (:page-size pool)
                  :physical-pages (:physical-pages pool)
                  :scale (or scale (/ 1.0 (Math/sqrt (double qk-head-dim))))
                  :q-dtype query-dtype
                  :k-dtype :half
                  :v-dtype :half
                  :output-dtype output-dtype
                  :accumulator-dtype :float
                  :k-layout :page-major
                  :v-layout :page-major
                  :visibility (attention/visibility {:causal? true})})
        routed (attention-route/route! problem)
        specs (attention/buffer-specs problem)
        _ (checked-resident-view :query query-view)
        _ (checked-resident-view :output output-view)
        allocations
        (cond->
         {(:query-row-offsets keys)
          [:int (get-in specs [(:query-row-offsets ids) :elements]) nil :input]
          (:query-positions keys)
          [:int (get-in specs [(:query-positions ids) :elements]) nil :input]
          (:page-table keys) [:int (get-in specs [(:page-table ids) :elements]) nil :input]
          (:lengths keys) [:int (get-in specs [(:lengths ids) :elements]) nil :input]
          (:start-positions keys)
          [:int (get-in specs [(:start-positions ids) :elements]) nil :input]}
          (nil? query-view)
          (assoc (:query keys)
                 [query-dtype (get-in specs [(:query ids) :elements]) nil :input])
          (nil? output-view)
          (assoc (:output keys)
                 [output-dtype (get-in specs [(:output ids) :elements]) nil :output]))
        bindings {(:query ids) (or query-view (:query keys))
                  (:query-row-offsets ids) (:query-row-offsets keys)
                  (:query-positions ids) (:query-positions keys)
                  (:key-pages ids) (get (page-pool/buffer-keys pool) [:key layer])
                  (:value-pages ids) (get (page-pool/buffer-keys pool) [:value layer])
                  (:page-table ids) (:page-table keys)
                  (:lengths ids) (:lengths keys)
                  (:start-positions ids) (:start-positions keys)
                  (:output ids) (or output-view (:output keys))}]
    {:problem problem
     :graph (:graph routed)
     :strategy (:strategy routed)
     :ids ids
     :buffer-keys keys
     :bindings bindings
     :allocations allocations
     :owned-buffer-keys (set (clojure.core/keys allocations))
     :options (assoc opts
                     :layer layer
                     :batch-size batch-size
                     :total-query-tokens total-query-tokens
                     :pages-per-sequence pages-per-sequence
                     :query-dtype query-dtype
                     :output-dtype output-dtype)}))

(defn open-runner!
  "Allocate and bind a fixed-capacity Raster paged-attention reference runner.

  The caller owns `pool`, its Raster session, and any `:query-view` or
  `:output-view`. Closing the runner releases its graph and private buffers, but
  not the page pool or caller-owned views. Raster validates view dtype, extent,
  allocation lifetime, and session identity while binding."
  [pool opts]
  (let [{:keys [problem graph buffer-keys bindings allocations] :as plan}
        (reference-plan pool opts)
        session (:session pool)
        graph-key (or (:graph-key opts)
                      [:paged-attention (:id problem)])]
    (try
      (gpu/alloc! session allocations)
      (let [handle (gpu/bind-kernel-graph! session graph-key graph bindings {})]
        (map->PagedAttentionRunner
         {:session session
          :pool pool
          :problem problem
          :handle handle
          :buffer-keys buffer-keys
          :graph-key graph-key
          :state (atom {:closed? false :pending nil :lease nil :plan plan})}))
      (catch Throwable error
        (doseq [key (keys allocations)]
          (when (gpu/buffer session key)
            (gpu/free-buffer! session key)))
        (throw error)))))

(defn- tensor-array
  [dtype values expected]
  (let [result
        (case dtype
          :half
          (cond
            (instance? (Class/forName "[S") values) values
            (or (instance? (Class/forName "[F") values) (sequential? values))
            (short-array (map #(Float/floatToFloat16 (float %)) values))
            :else
            (throw (ex-info "FP16 query values must be half bits or numeric values"
                            {:value-type (type values)})))

          :float
          (cond
            (instance? (Class/forName "[F") values) values
            (sequential? values) (float-array values)
            :else
            (throw (ex-info "FP32 query values must be floats or numeric values"
                            {:value-type (type values)}))))]
    (when-not (= expected (alength result))
      (throw (ex-info "Query payload has the wrong element count"
                      {:expected expected :actual (alength result)})))
    result))

(defn load-batch!
  "Validate and install one packed query batch and its continuation routes.

  `batch` contains `:continuation-ids`, `:query-values`, `:row-offsets`, and
  `:positions`. Optional `:append-reservations` must be aligned with the
  continuation identities and loads prospective post-append routes. This is
  safe when append and attention are submitted in that order on Raster's
  in-order session queue. Omit `:query-values` when the runner was opened with
  `:query-view`; the producer must have populated that resident view before
  submission. Its capacities must exactly match the runner. No graph may be in
  flight while these reusable descriptor buffers are changed. Returns `runner`."
  [runner {:keys [continuation-ids query-values row-offsets positions
                  append-reservations]}]
  (locking runner
    (let [{:keys [closed? pending lease plan]} @(:state runner)
          {:keys [batch-size total-query-tokens pages-per-sequence query-dtype]}
          (:options plan)
          problem (:problem runner)
          resident-query? (some? (get-in plan [:options :query-view]))]
      (when closed?
        (throw (ex-info "Paged-attention runner is closed" {})))
      (when pending
        (throw (ex-info "Cannot replace a batch while its graph is in flight"
                        {:event pending})))
      (when-not (= batch-size (count continuation-ids))
        (throw (ex-info "Continuation batch has the wrong lane count"
                        {:expected batch-size :actual (count continuation-ids)})))
      (when (and append-reservations
                 (not= (vec continuation-ids)
                       (mapv :continuation-id append-reservations)))
        (throw (ex-info "Append reservations do not align with the attention batch"
                        {:continuation-ids (vec continuation-ids)
                         :reservation-ids
                         (mapv :continuation-id append-reservations)})))
      (when (= resident-query? (some? query-values))
        (throw (ex-info (if resident-query?
                          "Resident-query runner does not accept host query values"
                          "Private-query runner requires host query values")
                        {:resident-query? resident-query?})))
      (let [new-lease (if append-reservations
                        (page-pool/acquire-prospective-lease!
                         (:pool runner) append-reservations)
                        (page-pool/acquire-lease!
                         (:pool runner) continuation-ids))
            offsets (int-array row-offsets)
            positions (int-array positions)
            route-values (page-pool/leased-dense-route-values
                          (:pool runner) new-lease
                          {:pages-per-sequence pages-per-sequence})
            q-elements (* total-query-tokens (:q-heads problem)
                          (:qk-head-dim problem))
            query-values (when-not resident-query?
                           (tensor-array (or query-dtype :half)
                                         query-values q-elements))
            keys (:buffer-keys runner)]
        (try
          (attention/validate-query-values! problem offsets positions)
          (attention/validate-routing! problem route-values)
          (gpu/upload-ranges!
           (:session runner)
           (cond->
            []
             (not resident-query?)
             (conj [(:query keys) query-values
                    {:elements (alength query-values)}])
             true
             (into [[(:query-row-offsets keys) offsets {:elements (alength offsets)}]
                    [(:query-positions keys) positions {:elements (alength positions)}]
                    [(:page-table keys) (:page-table route-values)
                     {:elements (alength ^ints (:page-table route-values))}]
                    [(:lengths keys) (:lengths route-values)
                     {:elements (alength ^ints (:lengths route-values))}]
                    [(:start-positions keys) (:start-positions route-values)
                     {:elements (alength ^ints (:start-positions route-values))}]])))
          (swap! (:state runner) assoc :lease new-lease)
          (when lease
            (page-pool/release-lease! (:pool runner) lease))
          runner
          (catch Throwable error
            (page-pool/release-lease! (:pool runner) new-lease)
            (throw error)))))))

(defn submit!
  "Submit the loaded batch without blocking and return Raster's GPU event."
  [runner]
  (locking runner
    (let [{:keys [closed? pending]} @(:state runner)]
      (when closed?
        (throw (ex-info "Paged-attention runner is closed" {})))
      (when pending
        (throw (ex-info "Paged-attention runner already has an in-flight graph"
                        {:event pending})))
      (when-not (:lease @(:state runner))
        (throw (ex-info "Paged-attention runner has no loaded batch" {})))
      (let [event (gpu/submit-kernel-graph! (:session runner) (:handle runner))]
        (swap! (:state runner) assoc :pending event)
        event))))

(defn await!
  "Wait for `event`, release it, and return the attention result.

  A private-output runner downloads and returns the configured output array
  (FP16 bits or FP32 values). A runner opened with `:output-view` returns that
  resident view after establishing GPU completion; no tensor data crosses to
  the host."
  [runner event]
  (locking runner
    (when-not (= event (:pending @(:state runner)))
      (throw (ex-info "GPU event does not belong to this runner submission"
                      {:expected (:pending @(:state runner)) :actual event})))
    (let [completed? (atom false)]
      (try
        (gpu/await-event! (:session runner) event)
        (reset! completed? true)
        (or (get-in @(:state runner) [:plan :options :output-view])
            (gpu/download (:session runner) (:output (:buffer-keys runner))))
        (finally
          (gpu/release-event! (:session runner) event)
          (when @completed?
            (when-let [lease (:lease @(:state runner))]
              (page-pool/release-lease! (:pool runner) lease)))
          (swap! (:state runner) assoc :pending nil
                 :lease (if @completed? nil (:lease @(:state runner)))))))))

(defn run!
  "Load, execute, and synchronously return one batch's attention result.

  The return shape follows `await!`: a typed host array for private output, or
  the caller-owned resident output view when one was configured."
  [runner batch]
  (load-batch! runner batch)
  (await! runner (submit! runner)))

(defn close-runner!
  "Wait for pending work and release runner-owned graph and buffers. Idempotent."
  [runner]
  (locking runner
    (when-not (:closed? @(:state runner))
      (try
        (when-let [event (:pending @(:state runner))]
          (gpu/release-event! (:session runner) event))
        (gpu/release-kernel-graph! (:session runner) (:handle runner))
        (finally
          (when-let [lease (:lease @(:state runner))]
            (page-pool/release-lease! (:pool runner) lease))
          (doseq [key (get-in @(:state runner) [:plan :owned-buffer-keys])]
            (when (gpu/buffer (:session runner) key)
              (gpu/free-buffer! (:session runner) key)))
          (swap! (:state runner) assoc :closed? true :pending nil :lease nil)))))
  nil)
