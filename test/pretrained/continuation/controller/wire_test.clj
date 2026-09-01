(ns pretrained.continuation.controller.wire-test
  (:require [clojure.test :refer [deftest is testing]]
            [pretrained.continuation.controller.wire :as wire]))

(deftest offer-round-trips-through-worker-addressing
  (let [effect {:effect/op :router/send-offer
                :effect/to :worker-a
                :assignment/id [:request-a 1]
                :assignment/worker-epoch 3
                :assignment/request {:request/id :request-a}
                :assignment/candidate {:candidate/worker-id :worker-a}}
        message (wire/effect->message effect)]
    (is (wire/control-message? message))
    (is (= :continuation/offer (:type message)))
    (is (nil? (wire/worker-event :worker-b message)))
    (is (= (-> effect
               (dissoc :effect/op :effect/to)
               (assoc :event/type :assignment/offered))
           (wire/worker-event :worker-a message)))))

(deftest worker-results-round-trip-to-router
  (doseq [[op type event-type]
          [[:worker/send-offer-result
            :continuation/offer-result :worker/offer-result]
           [:worker/send-result :continuation/result :worker/result]]]
    (let [effect {:effect/op op :effect/to :router
                  :request/id :request-a
                  :assignment/id [:request-a 1]
                  :event/result {:status :completed}}
          message (wire/effect->message effect)]
      (is (= type (:type message)))
      (is (= event-type (:event/type (wire/router-event message)))))))

(deftest local-effects-cannot-leak-to-kabel
  (testing "GPU operations and timers are intentionally not serializable here"
    (is (thrown-with-msg?
         clojure.lang.ExceptionInfo #"local"
         (wire/effect->message {:effect/op :worker/decode})))
    (is (thrown-with-msg?
         clojure.lang.ExceptionInfo #"local"
         (wire/effect->message {:effect/op :router/set-offer-timer})))))
