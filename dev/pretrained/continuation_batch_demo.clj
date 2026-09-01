(ns pretrained.continuation-batch-demo
  "Model-free REPL demonstration of the paged multi-request worker loop."
  (:require [pretrained.continuation.paged-runtime :as paged-runtime]))

(defn- effect
  [id prompt max-new]
  {:assignment/id [id 1]
   :assignment/request
   {:request/id id
    :request/continuation-id id
    :request/model-fingerprint "batch-demo-model-v1"
    :request/tokens prompt
    :request/max-new-tokens max-new}})

(defn run-simulation
  "Run three variable-length prompt continuations through four fixed lanes.

  Decoder calls are simulated, but queueing, prefill reservation, selective
  lane priming, lane retention/refill, and completion use the production
  runtime. Returns compact scheduling evidence suitable for REPL inspection."
  []
  (let [route-counts (atom {})
        submissions (atom [])
        primes (atom [])
        runtime
        (paged-runtime/open-runtime
         {:decode-state {:batch-size 4 :maxpos 128}}
         {:prime-lanes! (fn [_ values] (swap! primes conj values))
          :step-lanes!
          (fn [_ work]
            (Thread/sleep 1)
            (swap! submissions conj work)
            (mapv (fn [{:keys [continuation-id position] :as item}]
                    (swap! route-counts update continuation-id (fnil inc 0))
                    (assoc item :token (+ 1000 position)))
                  work))
          :route-token-count (fn [_ id] (get @route-counts id 0))})
        requests [(effect :short (vec (range 5)) 3)
                  (effect :medium (vec (range 9)) 2)
                  (effect :long (vec (range 13)) 4)]]
    (try
      (let [tasks
            (mapv (fn [request]
                    (future
                      (paged-runtime/prefill! runtime request)
                      (paged-runtime/decode! runtime request)))
                  requests)
            results (mapv deref tasks)
            widths (mapv count @submissions)]
        {:outputs (into {}
                        (map (fn [request result]
                               [(get-in request
                                        [:assignment/request :request/id])
                                (:tokens result)])
                             requests results))
         :iterations (count @submissions)
         :maximum-active-lanes (reduce max 0 widths)
         :multi-request-iterations (count (filter #(> % 1) widths))
         :selective-prime-uploads (count @primes)
         :runtime (paged-runtime/state runtime)})
      (finally
        (.close runtime)))))
