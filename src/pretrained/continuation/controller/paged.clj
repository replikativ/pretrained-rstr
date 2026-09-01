(ns pretrained.continuation.controller.paged
  "Paged-decoder operation handlers for a worker-local controller."
  (:require [pretrained.continuation.manager :as manager]
            [pretrained.continuation.page-pool :as page-pool]
            [pretrained.continuation.paged-decoder :as paged-decoder]))

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
          :policy policy})]
    {:ok? true
     :cached-token-count (:cached-token-count result)}))

(defn- prefill-suffix!
  [decoder effect]
  (let [request (:assignment/request effect)]
    (paged-decoder/prime-prompt!
     decoder (continuation-id effect) (:request/tokens request))
    {:ok? true}))

(defn- decode!
  [decoder eos-ids effect]
  (let [request (:assignment/request effect)
        id (continuation-id effect)
        prompt (:request/tokens request)
        max-new (:request/max-new-tokens request)
        start-position (dec (count prompt))]
    ;; This is deliberately idempotent after the prefill handler. It also
    ;; repairs a stale lower-tier observation that restored fewer exact tokens
    ;; than the candidate advertised, while never recomputing resident rows.
    (paged-decoder/prime-prompt! decoder id prompt)
    (loop [position start-position
           output []]
      (if (and (< (count output) max-new)
               (< (:token-count (page-pool/route (:pool decoder) id))
                  (long (get-in decoder [:decode-state :maxpos]))))
        (let [token (paged-decoder/step! decoder id position)
              output (conj output token)]
          (if (contains? eos-ids token)
            {:ok? true :tokens output}
            (recur (inc position) output)))
        {:ok? true :tokens output}))))

(defn handlers
  "Return local-controller handlers for `cache` and a paged `decoder`.

  Options accept `:eos-ids` and resident `:policy`. The restore handler consumes
  the capacity reservation created before offer acceptance. Prefill commits
  only the exact uncached suffix; decode uses the existing Raster linked paged
  executable and leaves the completed route resident for reuse or checkpoint."
  ([cache decoder] (handlers cache decoder {}))
  ([cache decoder {:keys [eos-ids policy]
                   :or {eos-ids #{} policy {:durable? false}}}]
   {:worker/restore-prefix #(restore-prefix! cache decoder policy %)
    :worker/prefill-suffix #(prefill-suffix! decoder %)
    :worker/decode #(decode! decoder eos-ids %)}))
