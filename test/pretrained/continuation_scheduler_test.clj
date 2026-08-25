(ns pretrained.continuation-scheduler-test
  (:require [clojure.test :refer [deftest is]]
            [pretrained.continuation.scheduler :as scheduler]))

(deftest decode-work-precedes-bounded-prefill
  (let [requests [(scheduler/request
                   #:request{:id :long-prompt :phase :prefill
                             :remaining-tokens 500 :arrival 0})
                  (scheduler/request
                   #:request{:id :decode-b :phase :decode
                             :remaining-tokens 1 :arrival 2})
                  (scheduler/request
                   #:request{:id :decode-a :phase :decode
                             :remaining-tokens 1 :arrival 1})]
        plan (scheduler/plan-iteration
              {:max-batched-tokens 66 :max-sequences 3
               :prefill-chunk-size 64}
              requests)]
    (is (= [:decode-a :decode-b :long-prompt]
           (mapv :request/id (:scheduled plan))))
    (is (= [1 1 64] (mapv :scheduled/tokens (:scheduled plan))))
    (is (zero? (:unused-token-capacity plan)))
    (is (empty? (:deferred plan)))))

(deftest iteration-capacity-defers-without-mutating-requests
  (let [requests [#:request{:id :a :phase :decode :remaining-tokens 4 :arrival 0}
                  #:request{:id :b :phase :decode :remaining-tokens 2 :arrival 1}]
        plan (scheduler/plan-iteration
              {:max-batched-tokens 1 :max-sequences 1} requests)]
    (is (= [:a] (mapv :request/id (:scheduled plan))))
    (is (= [:b] (mapv :request/id (:deferred plan))))
    (is (= 3 (:request/remaining-tokens
              (scheduler/advance-request (first (:scheduled plan))))))
    (is (= 4 (:request/remaining-tokens (first requests))))))

(deftest optional-prefill-reservation-prevents-starvation
  (let [requests (conj (mapv (fn [id]
                               #:request{:id id :phase :decode
                                         :remaining-tokens 8})
                             (range 4))
                       #:request{:id :waiting-prefill :phase :prefill
                                 :remaining-tokens 100})
        plan (scheduler/plan-iteration
              {:max-batched-tokens 4 :max-sequences 4
               :minimum-prefill-tokens 1 :prefill-chunk-size 16}
              requests)]
    (is (= [0 1 2 :waiting-prefill]
           (mapv :request/id (:scheduled plan))))
    (is (= [1 1 1 1] (mapv :scheduled/tokens (:scheduled plan))))
    (is (= [3] (mapv :request/id (:deferred plan))))))

(deftest exact-cache-source-is-default-and-approximation-is-explicit
  (let [candidates [#:source{:kind :recompute-exact :estimated-ms 12.0}
                    #:source{:kind :restore-exact :estimated-ms 8.0}
                    #:source{:kind :modular-repair :estimated-ms 3.0 :quality 0.97}]]
    (is (= :restore-exact
           (:source/kind (scheduler/choose-cache-source candidates))))
    (is (= :modular-repair
           (:source/kind
            (scheduler/choose-cache-source
             candidates {:allow-approximate? true :minimum-quality 0.95}))))
    (is (= :restore-exact
           (:source/kind
            (scheduler/choose-cache-source
             candidates {:allow-approximate? true :minimum-quality 0.99}))))))
