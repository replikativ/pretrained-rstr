(ns pretrained.continuation.controller.paged
  "Paged-decoder operation handlers for a worker-local controller."
  (:require [pretrained.continuation.chunk :as chunk]
            [pretrained.continuation.manager :as manager]
            [pretrained.continuation.page-pool :as page-pool]
            [pretrained.continuation.paged-decoder :as paged-decoder]
            [pretrained.continuation.paged-runtime :as paged-runtime]))

(defn- continuation-id
  [effect]
  (let [request (:assignment/request effect)]
    (or (:request/continuation-id request) (:request/id request))))

(defn- restore-prefix!
  [cache decoder policy effect]
  (let [request (:assignment/request effect)
        result
        (manager/restore-paged-prefix!
         cache (:pool decoder) (continuation-id effect)
         (:request/model-fingerprint request) (:request/tokens request)
         {:capacity-reservation (:worker/capacity-reservation effect)
          :maximum-cached-token-count
          (get-in effect [:assignment/candidate :estimate/cached-token-count])
          :policy policy})]
    {:ok? true
     :cached-token-count (:cached-token-count result)}))

(defn- restore-prefix-overlapped!
  [runtime cache decoder policy effect]
  (let [request (:assignment/request effect)
        result
        (paged-runtime/run-background-operation!
         runtime (:assignment/id effect)
         (fn [cancelled?]
           (manager/restore-paged-prefix-overlapped!
            cache (:pool decoder) (continuation-id effect)
            (:request/model-fingerprint request) (:request/tokens request)
            {:capacity-reservation (:worker/capacity-reservation effect)
             :maximum-cached-token-count
             (get-in effect [:assignment/candidate
                             :estimate/cached-token-count])
             :policy policy
             :cancelled? cancelled?})))]
    {:ok? true :cached-token-count (:cached-token-count result)}))

(defn- prefill-suffix!
  [decoder effect]
  (let [request (:assignment/request effect)]
    (paged-decoder/prime-prompt!
     decoder (continuation-id effect) (:request/tokens request))
    {:ok? true}))

(defn- ensure-route!
  [decoder policy effect]
  (let [id (continuation-id effect)
        pool (:pool decoder)]
    (or (page-pool/route pool id)
        (page-pool/allocate-route!
         pool id 0
         {:policy policy
          :capacity-reservation (:worker/capacity-reservation effect)}))))

(defn- mark-exact-prefix!
  [decoder id request output chunk-size policy]
  (let [pool (:pool decoder)
        resident-route (page-pool/route pool id)
        processed (:token-count resident-route)
        history (into (vec (:request/tokens request)) output)
        tail (peek (chunk/plan history processed chunk-size))]
    (when tail
      (page-pool/touch-route!
       pool id
       (merge policy
              {:model-fingerprint (:request/model-fingerprint request)
               :prefix-hash (:chunk/prefix-hash tail)
               :bytes (page-pool/route-bytes pool id)})))))

(defn- decode!
  [decoder eos-ids chunk-size policy effect]
  (let [request (:assignment/request effect)
        id (continuation-id effect)
        prompt (:request/tokens request)
        max-new (:request/max-new-tokens request)
        start-position (dec (count prompt))]
    ;; This is deliberately idempotent after the prefill handler. It also
    ;; repairs a stale lower-tier observation that restored fewer exact tokens
    ;; than the candidate advertised, while never recomputing resident rows.
    (paged-decoder/prime-prompt! decoder id prompt)
    (let [output
          (loop [position start-position
                 output []]
            (if (and (< (count output) max-new)
                     (< (:token-count (page-pool/route (:pool decoder) id))
                        (long (get-in decoder [:decode-state :maxpos]))))
              (let [token (paged-decoder/step! decoder id position)
                    output (conj output token)]
                (if (contains? eos-ids token)
                  output
                  (recur (inc position) output)))
              output))]
      (mark-exact-prefix! decoder id request output chunk-size policy)
      {:ok? true :tokens output})))

(defn handlers
  "Return local-controller handlers for `cache` and a paged `decoder`.

  Options accept `:eos-ids`, resident `:policy`, and `:chunk-size` (defaulting to
  the manager's chunk size). The restore handler consumes the capacity
  reservation created before offer acceptance. Prefill commits only the exact
  uncached suffix; decode uses the existing Raster linked paged executable and
  annotates the completed route with its exact prefix identity for worker
  observations, reuse, or checkpoint."
  ([cache decoder] (handlers cache decoder {}))
  ([cache decoder {:keys [eos-ids policy chunk-size]
                   :or {eos-ids #{} policy {:durable? false}}}]
   (let [chunk-size (or chunk-size (:chunk-size cache)
                        chunk/default-chunk-size)]
     {:worker/restore-prefix #(restore-prefix! cache decoder policy %)
      :worker/prefill-suffix #(prefill-suffix! decoder %)
      :worker/decode #(decode! decoder eos-ids chunk-size policy %)})))

(defn batched-handlers
  "Return controller handlers backed by a shared paged batch runtime.

  Storage localization and retained GPU uploads poll outside the decoder loop,
  so unrelated lanes continue while a cached prefix becomes resident. Missing
  prompt suffixes and generation then run incrementally in sparse fixed lanes.
  The local controller
  must use concurrent submission callbacks such as
  `paged-runtime/controller-submission`; its single-thread default cannot place
  multiple blocking handler jobs into one batch. The caller owns `runtime`."
  ([runtime cache decoder] (batched-handlers runtime cache decoder {}))
  ([runtime cache decoder {:keys [policy chunk-size]
                           :or {policy {:durable? false}}}]
   (when-not (identical? decoder (:decoder runtime))
     (throw (ex-info "Paged runtime belongs to a different decoder" {})))
   (let [chunk-size (or chunk-size (:chunk-size cache)
                        chunk/default-chunk-size)]
     {:worker/restore-prefix
      #(restore-prefix-overlapped! runtime cache decoder policy %)
      :worker/prefill-suffix
      (fn [effect]
        (paged-runtime/run-operation!
         runtime (:assignment/id effect)
         #(ensure-route! decoder policy effect))
        (paged-runtime/prefill! runtime effect))
      :worker/decode
      (fn [effect]
        (paged-runtime/run-operation!
         runtime (:assignment/id effect)
         #(ensure-route! decoder policy effect))
        (let [result (paged-runtime/decode! runtime effect)
              request (:assignment/request effect)
              id (continuation-id effect)]
          (mark-exact-prefix! decoder id request (:tokens result)
                              chunk-size policy)
          result))})))
