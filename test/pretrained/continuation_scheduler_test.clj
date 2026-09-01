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

(deftest decode-lanes-retain-resident-work-and-refill-only-vacancies
  (let [left #:request{:id :left :continuation-id :left-kv :phase :decode
                       :remaining-tokens 4 :position 8 :pending-token 101 :arrival 0}
        finished #:request{:id :finished :continuation-id :finished-kv :phase :decode
                           :remaining-tokens 1 :position 5 :pending-token 102 :arrival 1}
        incoming #:request{:id :incoming :continuation-id :incoming-kv :phase :decode
                           :remaining-tokens 3 :position 2 :pending-token 103 :arrival 2}
        plan (scheduler/plan-decode-lanes
              3 [left finished nil] [incoming left])
        submission (scheduler/decode-submission plan)]
    (is (= [:left :incoming nil] (mapv :request/id (:lanes plan))))
    (is (= [:left] (mapv :request/id (:retained plan))))
    (is (= [:incoming] (mapv :request/id (:refill plan))))
    (is (= [:finished] (mapv :request/id (:retired plan))))
    (is (= #{:left-kv :incoming-kv} (:protected-continuation-ids plan)))
    (is (= [{:lane 0 :continuation-id :left-kv :position 8}
            {:lane 1 :continuation-id :incoming-kv :position 2}]
           (:lane-work submission)))
    (is (= [{:lane 1 :token 103}] (:prime-lanes submission))
        "the retained lane keeps the embedding produced by the prior tail")))

(deftest decode-lane-overflow-is-deferred-with-stable-priority
  (let [requests [#:request{:id :later :phase :decode :remaining-tokens 1 :arrival 2}
                  #:request{:id :first :phase :decode :remaining-tokens 1 :arrival 0}
                  #:request{:id :middle :phase :decode :remaining-tokens 1 :arrival 1}]
        plan (scheduler/plan-decode-lanes 2 [] requests)]
    (is (= [:first :middle] (mapv :request/id (:lanes plan))))
    (is (= [:later] (mapv :request/id (:deferred plan))))))

(deftest mixed-work-lanes-follow-iteration-order-with-stable-refill
  (let [decode #:request{:id :decode :continuation-id :decode-kv
                         :phase :decode :remaining-tokens 4}
        old-prefill #:request{:id :old-prefill :continuation-id :old-kv
                              :phase :prefill :remaining-tokens 8}
        prefill #:request{:id :prefill :continuation-id :prefill-kv
                          :phase :prefill :remaining-tokens 7}
        plan (scheduler/plan-work-lanes
              3 [decode old-prefill nil] [prefill decode])]
    (is (= [:decode :prefill nil] (mapv :request/id (:lanes plan))))
    (is (= [:decode] (mapv :request/id (:retained plan))))
    (is (= [:prefill] (mapv :request/id (:refill plan))))
    (is (= [:old-prefill] (mapv :request/id (:retired plan))))
    (is (= #{:decode-kv :prefill-kv}
           (:protected-continuation-ids plan)))))

(deftest decode-results-retire-eos-and-preserve-next-token-state
  (let [requests [#:request{:id :keep :continuation-id :keep-kv :phase :decode
                            :remaining-tokens 3 :position 7 :pending-token 10}
                  #:request{:id :stop :continuation-id :stop-kv :phase :decode
                            :remaining-tokens 3 :position 4 :pending-token 11}]
        plan (scheduler/plan-decode-lanes 3 [] requests)
        result (scheduler/complete-decode-iteration
                plan [{:lane 0 :token 20} {:lane 1 :token 99}]
                {:eos-ids #{99}})]
    (is (= [:keep nil nil] (mapv :request/id (:lanes result))))
    (is (= 8 (get-in result [:lanes 0 :request/position])))
    (is (= 20 (get-in result [:lanes 0 :request/pending-token])))
    (is (= 2 (get-in result [:lanes 0 :request/remaining-tokens])))
    (is (= [:keep] (mapv :request/id (:runnable result))))
    (is (= [:stop] (mapv :request/id (:completed result))))
    (is (= 99 (get-in result [:completed 0 :iteration/token])))))
