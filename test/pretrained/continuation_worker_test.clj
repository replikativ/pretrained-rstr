(ns pretrained.continuation-worker-test
  (:require [clojure.test :refer [deftest is]]
            [pretrained.continuation.paged-decoder :as paged-decoder]
            [pretrained.continuation.worker :as worker]))

(defn- decode-request
  [id token position remaining arrival]
  #:request{:id id
            :continuation-id (keyword (str (name id) "-kv"))
            :phase :decode
            :remaining-tokens remaining
            :pending-token token
            :position position
            :arrival arrival})

(deftest worker-retains-refills-executes-and-retires-lanes
  (let [decoder {:decode-state {:batch-size 3}}
        retained (decode-request :retained 10 7 3 0)
        departed (decode-request :departed 11 4 2 1)
        incoming (decode-request :incoming 12 2 4 2)
        calls (atom [])]
    (with-redefs [paged-decoder/prime-lanes!
                  (fn [engine rows]
                    (swap! calls conj [:prime rows])
                    engine)
                  paged-decoder/step-lanes!
                  (fn [_ lane-work]
                    (swap! calls conj [:step lane-work])
                    [{:lane 0 :continuation-id :retained-kv :position 7 :token 20}
                     {:lane 1 :continuation-id :incoming-kv :position 2 :token 99}])]
      (let [iteration
            (worker/run-decode-iteration!
             decoder [retained departed nil] [incoming retained] :eos-ids #{99})]
        (is (= [[:prime [{:lane 1 :token 12}]]
                [:step [{:lane 0 :continuation-id :retained-kv :position 7}
                        {:lane 1 :continuation-id :incoming-kv :position 2}]]]
               @calls))
        (is (= [:retained nil nil] (mapv :request/id (:lanes iteration))))
        (is (= [:incoming] (mapv :request/id (:completed iteration))))
        (is (= [:departed] (mapv :request/id (:retired iteration))))
        (is (= #{:retained-kv :incoming-kv}
               (:protected-continuation-ids iteration)))
        (is (= #{:retained-kv} (:next-protected-continuation-ids iteration)))
        (is (= [20 99] (mapv :token (:results iteration))))))))

(deftest empty-worker-iteration-submits-no-device-work
  (let [calls (atom [])]
    (with-redefs [paged-decoder/prime-lanes! #(swap! calls conj [:prime %1 %2])
                  paged-decoder/step-lanes! #(swap! calls conj [:step %1 %2])]
      (let [iteration (worker/run-decode-iteration!
                       {:decode-state {:batch-size 2}} [] [])]
        (is (empty? @calls))
        (is (= [nil nil] (:lanes iteration)))
        (is (empty? (:results iteration)))))))

(deftest next-ready-boundary-keeps-policy-coordination-explicit
  (let [iteration {:runnable [(decode-request :running 1 1 2 0)]
                   :deferred [(decode-request :waiting 2 2 3 1)]}
        arrivals [(decode-request :new 3 0 4 2)]]
    (is (= [:running :waiting :new]
           (mapv :request/id (worker/next-ready-requests iteration arrivals))))))

(deftest continuous-generation-retires-eos-lanes-without-stopping-the-batch
  (let [decoder {:decode-state {:batch-size 2}}
        calls (atom [])
        emitted (atom [[10 99] [11] [12]])]
    (with-redefs [paged-decoder/prime-prompts-batch!
                  (fn [engine ids prompts]
                    (swap! calls conj [:prompts ids prompts])
                    engine)
                  paged-decoder/prime-lanes!
                  (fn [& args]
                    (swap! calls conj [:unexpected-prime args]))
                  paged-decoder/step-lanes!
                  (fn [_ lane-work]
                    (let [tokens (first @emitted)]
                      (swap! emitted next)
                      (swap! calls conj [:step lane-work])
                      (mapv #(assoc %1 :token %2) lane-work tokens)))]
      (is (= [[10 11 12] [99]]
             (worker/generate-continuously!
              decoder [:left :right] [[1 2] [3 4]] 3 :eos-ids #{99})))
      (is (= [[:prompts [:left :right] [[1 2] [3 4]]]
              [:step [{:lane 0 :continuation-id :left :position 1}
                      {:lane 1 :continuation-id :right :position 1}]]
              [:step [{:lane 0 :continuation-id :left :position 2}]]
              [:step [{:lane 0 :continuation-id :left :position 3}]]]
             @calls)))))
