(ns pretrained.continuation-transfer-demo
  "REPL-visible simulation of cache restore overlapping a hot decode lane."
  (:require [pretrained.continuation.paged-runtime :as paged-runtime]))

(defn run-simulation
  "Run one pending restore beside an unrelated two-token decode.

  The restore boundary is controlled by a promise rather than wall-clock sleep,
  making the ordering deterministic on a laptop. Returns the event order,
  decode output, and drained runtime state."
  []
  (let [route-counts (atom {:hot 2})
        events (atom [])
        restore-entered (promise)
        release-restore (promise)
        runtime
        (paged-runtime/open-runtime
         {:decode-state {:batch-size 2 :maxpos 64}}
         {:prime-lanes! (fn [& _] nil)
          :step-lanes!
          (fn [_ work]
            (mapv (fn [{:keys [continuation-id position] :as item}]
                    (swap! route-counts update continuation-id inc)
                    (assoc item :token (+ 1000 position)))
                  work))
          :route-token-count (fn [_ id] (get @route-counts id 0))})
        restore
        (future
          (paged-runtime/run-background-operation!
           runtime :cold-assignment
           (fn [_cancelled?]
             (swap! events conj :restore-submitted)
             (deliver restore-entered true)
             @release-restore
             (swap! events conj :restore-complete)
             {:ok? true})))]
    (try
      @restore-entered
      (let [decode
            (paged-runtime/decode!
             runtime
             {:assignment/id :hot-assignment
              :assignment/request
              {:request/id :hot-assignment
               :request/continuation-id :hot
               :request/model-fingerprint "fixture-model-v1"
               :request/tokens [7 8 9]
               :request/max-new-tokens 2}})]
        (swap! events conj :decode-complete)
        (deliver release-restore true)
        @restore
        {:events @events
         :decode decode
         :decode-overlapped-restore?
         (= [:restore-submitted :decode-complete :restore-complete] @events)
         :runtime (paged-runtime/state runtime)})
      (finally
        (deliver release-restore true)
        (.close runtime)))))
