(ns pretrained.continuation.controller.worker-test
  (:require [clojure.test :refer [deftest is]]
            [pretrained.continuation.controller.worker :as worker]))

(def request
  #:request{:id :r1 :model-fingerprint "gemma/test"
            :tokens (vec (range 33)) :max-new-tokens 8})

(def candidate
  #:candidate{:worker-id :w1 :worker-epoch 3 :cache-tier :ssd
              :cached-token-count 16 :cached-bytes 1024
              :queue-ms 0 :prefix-load-ms 2 :gpu-restore-ms 1
              :prefill-ms-per-token 0.5 :first-token-ms 1
              :page-size 8 :free-pages 8 :evictable-pages 0
              :max-context 128})

(def offer
  {:event/type :assignment/offered
   :assignment/id [:r1 1]
   :assignment/worker-epoch 3
   :assignment/request request
   :assignment/candidate candidate})

(deftest worker-rechecks-admission-and-fences-operation-results
  (let [initial (worker/initial-state
                 #:worker{:id :w1 :epoch 3 :models #{"gemma/test"}
                          :free-pages 8 :evictable-pages 0})
        {accepted :state effects :effects} (worker/transition initial offer)
        stale (worker/transition accepted
                                 {:event/type :worker/restore-result
                                  :assignment/id [:r1 0]
                                  :event/ok? true})
        {prefilling :state prefill-effects :effects}
        (worker/transition accepted {:event/type :worker/restore-result
                                     :assignment/id [:r1 1]
                                     :event/ok? true})
        {decoding :state decode-effects :effects}
        (worker/transition prefilling {:event/type :worker/prefill-result
                                       :assignment/id [:r1 1]
                                       :event/ok? true})
        completed
        (worker/transition decoding {:event/type :worker/decode-result
                                     :assignment/id [:r1 1]
                                     :event/ok? true
                                     :event/tokens [9 10]})]
    (is (= [:worker/send-offer-result :worker/restore-prefix]
           (mapv :effect/op effects)))
    (is (= accepted (:state stale)))
    (is (= [:worker/prefill-suffix] (mapv :effect/op prefill-effects)))
    (is (= [:worker/decode] (mapv :effect/op decode-effects)))
    (is (= :completed
           (get-in completed [:state :worker/assignments [:r1 1]
                              :assignment/phase])))
    (is (= [9 10]
           (get-in completed [:effects 0 :event/result :tokens])))))

(deftest stale-router-snapshot-cannot-overcommit-worker
  (let [initial (worker/initial-state
                 #:worker{:id :w1 :epoch 3 :models #{"gemma/test"}
                          :free-pages 1 :evictable-pages 0})
        result (worker/transition initial offer)]
    (is (false? (get-in result [:effects 0 :event/accepted?])))
    (is (= :insufficient-pages (get-in result [:effects 0 :event/reason])))
    (is (empty? (get-in result [:state :worker/assignments])))))

(deftest cancellation-releases-a-reservation-once
  (let [initial (worker/initial-state
                 #:worker{:id :w1 :epoch 3 :models #{"gemma/test"}
                          :free-pages 8 :evictable-pages 0})
        accepted (:state (worker/transition initial offer))
        cancelled (worker/transition accepted
                                     {:event/type :assignment/cancelled
                                      :assignment/id [:r1 1]})
        duplicate (worker/transition (:state cancelled)
                                     {:event/type :assignment/cancelled
                                      :assignment/id [:r1 1]})]
    (is (= 8 (get-in cancelled [:state :worker/free-pages])))
    (is (empty? (get-in cancelled [:state :worker/assignments])))
    (is (= [:worker/cancel-operation]
           (mapv :effect/op (:effects cancelled))))
    (is (empty? (:effects duplicate)))))
