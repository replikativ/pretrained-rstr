(ns pretrained.continuation.controller.router-test
  (:require [clojure.test :refer [deftest is testing]]
            [pretrained.continuation.controller.router :as router]))

(def request
  #:request{:id :r1
            :model-fingerprint "gemma/test"
            :tokens (vec (range 101))
            :max-new-tokens 16})

(defn- candidate
  [worker tier cached queue load restore prefill]
  #:candidate{:worker-id worker
              :worker-epoch 1
              :cache-tier tier
              :cached-token-count cached
              :cached-bytes (* cached 128)
              :queue-ms queue
              :prefix-load-ms load
              :gpu-restore-ms restore
              :prefill-ms-per-token prefill
              :first-token-ms 2.0
              :page-size 16
              :free-pages 32
              :evictable-pages 0
              :max-context 2048})

(deftest routing-minimizes-ready-to-first-token-not-prefix-length
  (let [remote (candidate :remote :object 100 0 80 8 0.5)
        local (candidate :local :gpu 64 5 0 0 0.5)
        ranked (router/rank-candidates request [remote local])]
    (is (= [:local :remote] (mapv :candidate/worker-id ranked)))
    (is (= 25.0 (:estimate/ttft-ms (first ranked))))
    (is (= 90.0 (:estimate/ttft-ms (second ranked))))))

(deftest queueing-and-prefix-loading-overlap
  (let [estimated (router/estimate-candidate
                   request (candidate :worker :ssd 80 30 20 5 1))]
    (is (= 57.0 (:estimate/ttft-ms estimated)))
    (is (= 20 (:estimate/missing-token-count estimated)))
    (is (= 8 (:estimate/required-pages estimated)))))

(deftest physical-transfer-capability-is-an-execution-policy-signal
  (let [eligible
        (router/estimate-candidate
         request
         (assoc (candidate :worker :ssd 80 30 20 5 1)
                :candidate/transfer-capabilities
                {:live-overlap-eligible? true}))
        conservative
        (router/estimate-candidate
         request (candidate :worker :ssd 80 30 20 5 1))]
    (is (true? (:estimate/live-transfer-overlap-eligible? eligible)))
    (is (false? (:estimate/live-transfer-overlap-eligible? conservative)))
    (is (= (:estimate/ttft-ms eligible) (:estimate/ttft-ms conservative))
        "capability is visible without inventing an unmeasured speedup")))

(deftest infeasible-capacity-and-context-are-declined
  (testing "generation growth participates in page admission"
    (let [small (assoc (candidate :small :none 0 0 0 0 1)
                       :candidate/free-pages 1
                       :candidate/evictable-pages 0)]
      (is (= :insufficient-pages
             (:estimate/reason (router/estimate-candidate request small))))))
  (testing "maximum context includes generated tokens"
    (let [short (assoc (candidate :short :none 0 0 0 0 1)
                       :candidate/max-context 110)]
      (is (= :context-capacity-exceeded
             (:estimate/reason (router/estimate-candidate request short)))))))

(deftest router-falls-back-and-fences-stale-results
  (let [{s1 :state fx1 :effects}
        (router/transition
         (router/initial-state)
         {:event/type :request/submitted
          :event/request request
          :event/candidates [(candidate :first :gpu 100 0 0 0 1)
                             (candidate :second :ssd 100 1 2 1 1)]})
        first-id (:assignment/id (first fx1))
        {s2 :state fx2 :effects}
        (router/transition s1 {:event/type :worker/offer-result
                               :request/id :r1
                               :assignment/id first-id
                               :event/accepted? false})
        second-id (:assignment/id (first fx2))
        {s3 :state}
        (router/transition s2 {:event/type :worker/offer-result
                               :request/id :r1
                               :assignment/id second-id
                               :event/accepted? true})
        stale (router/transition s3 {:event/type :worker/result
                                     :request/id :r1
                                     :assignment/id first-id
                                     :event/result {:status :completed :tokens [9]}})
        completed (router/transition s3 {:event/type :worker/result
                                         :request/id :r1
                                         :assignment/id second-id
                                         :event/result {:status :completed :tokens [7]}})]
    (is (= :first (:effect/to (first fx1))))
    (is (= :second (:effect/to (first fx2))))
    (is (not= first-id second-id))
    (is (= s3 (:state stale)))
    (is (= :completed
           (get-in completed [:state :router/requests :r1 :assignment/phase])))
    (is (= [7] (get-in completed [:effects 0 :response/value :tokens])))))
