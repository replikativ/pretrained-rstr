(ns pretrained.continuation.controller.cluster-test
  (:require [clojure.test :refer [deftest is]]
            [pretrained.continuation.controller.cluster :as cluster]))

(def ^:private request
  {:request/id :request-a
   :request/model-fingerprint "fixture-model-v1"
   :request/tokens [1 2 3]
   :request/max-new-tokens 2})

(defn- candidate
  [worker-id ttft]
  {:candidate/worker-id worker-id
   :candidate/worker-epoch 0
   :candidate/cache-tier :none
   :candidate/cached-token-count 0
   :candidate/cached-bytes 0
   :candidate/queue-ms 0
   :candidate/prefix-load-ms 0
   :candidate/gpu-restore-ms 0
   :candidate/prefill-ms-per-token (/ ttft 2)
   :candidate/first-token-ms 0
   :candidate/page-size 2
   :candidate/free-pages 8
   :candidate/evictable-pages 0
   :candidate/max-context 32})

(defn- fixture-controller
  [sent delivered timers]
  (cluster/open-controller
   {:send! #(swap! sent conj %)
    :deliver! #(swap! delivered conj %)
    :offer-timeout-ms 10
    :schedule! (fn [_ task]
                 (let [id (random-uuid)]
                   (swap! timers assoc id task)
                   id))
    :cancel-timer! #(swap! timers dissoc %)
    :close-scheduler! (constantly nil)}))

(deftest accepted-result-is-fenced-and-delivered
  (let [sent (atom []) delivered (atom []) timers (atom {})
        controller (fixture-controller sent delivered timers)]
    (try
      (cluster/submit! controller request
                       [(candidate :slow 4) (candidate :fast 2)])
      (let [offer (first @sent)]
        (is (= :fast (:effect/to offer)))
        (cluster/handle-event!
         controller
         {:event/type :worker/offer-result
          :request/id :request-a
          :assignment/id (:assignment/id offer)
          :event/accepted? true})
        (is (empty? @timers))
        (cluster/handle-event!
         controller
         {:event/type :worker/result
          :request/id :request-a
          :assignment/id [:request-a 999]
          :event/result {:status :completed :tokens [0]}})
        (is (empty? @delivered))
        (cluster/handle-event!
         controller
         {:event/type :worker/result
          :request/id :request-a
          :assignment/id (:assignment/id offer)
          :event/result {:status :completed :tokens [7 8]}})
        (is (= {:status :completed :tokens [7 8]}
               (:response/value (first @delivered)))))
      (finally
        (.close controller)))))

(deftest offer-timeout-falls-back-to-the-next-worker
  (let [sent (atom []) delivered (atom []) timers (atom {})
        controller (fixture-controller sent delivered timers)]
    (try
      (cluster/submit! controller request
                       [(candidate :first 1) (candidate :second 2)])
      ((first (vals @timers)))
      (is (= [:first :second] (mapv :effect/to @sent)))
      (is (= [:request-a 2] (:assignment/id (last @sent))))
      (finally
        (.close controller)))))
