(ns pretrained.continuation.controller.paged-test
  (:require [clojure.test :refer [deftest is]]
            [pretrained.continuation.controller.paged :as paged]
            [pretrained.continuation.manager :as manager]
            [pretrained.continuation.page-pool :as page-pool]
            [pretrained.continuation.paged-decoder :as paged-decoder]
            [pretrained.continuation.paged-runtime :as paged-runtime])
  (:import [java.util.concurrent CompletableFuture]))

(def ^:private request
  {:request/id :request-a
   :request/continuation-id :continuation-a
   :request/model-fingerprint "fixture-v1"
   :request/tokens [1 2 3]
   :request/max-new-tokens 3})

(defn- effect
  [op]
  {:effect/op op
   :assignment/id [:request-a 1]
   :assignment/request request
   :assignment/candidate {:estimate/cached-token-count 2}
   :worker/capacity-reservation ::capacity})

(deftest restore-passes-the-authoritative-capacity-reservation
  (let [captured (atom nil)
        handlers (paged/handlers ::cache {:pool ::pool}
                                {:policy {:durable? true}})]
    (with-redefs [manager/restore-paged-prefix!
                  (fn [& args]
                    (reset! captured args)
                    {:cached-token-count 2})]
      (is (= {:ok? true :cached-token-count 2}
             ((:worker/restore-prefix handlers)
              (effect :worker/restore-prefix))))
      (is (= [::cache ::pool :continuation-a "fixture-v1" [1 2 3]
              {:capacity-reservation ::capacity
               :maximum-cached-token-count 2
               :policy {:durable? true}}]
             @captured)))))

(deftest decode-rechecks-prefix-and-stops-at-eos
  (let [token-count (atom 2)
        primed (atom [])
        stepped (atom [])
        touched (atom nil)
        decoder {:pool ::pool :decode-state {:maxpos 8}}
        handlers (paged/handlers ::cache decoder {:eos-ids #{8}})]
    (with-redefs [paged-decoder/prime-prompt!
                  (fn [_ id prompt]
                    (swap! primed conj [id prompt]))
                  page-pool/route
                  (fn [_ _] {:token-count @token-count})
                  page-pool/route-bytes (fn [_ _] 128)
                  page-pool/touch-route!
                  (fn [_ id policy]
                    (reset! touched [id policy]))
                  paged-decoder/step!
                  (fn [_ id position]
                    (swap! token-count inc)
                    (swap! stepped conj [id position])
                    (if (= position 2) 7 8))]
      (is (= {:ok? true :tokens [7 8]}
             ((:worker/decode handlers) (effect :worker/decode))))
      (is (= [[:continuation-a [1 2 3]]] @primed))
      (is (= [[:continuation-a 2] [:continuation-a 3]] @stepped))
      (is (= :continuation-a (first @touched)))
      (is (= "fixture-v1" (:model-fingerprint (second @touched))))
      (is (uuid? (:prefix-hash (second @touched))))
      (is (= 128 (:bytes (second @touched)))))))

(deftest positive-value-decode-checkpoint-becomes-durable-after-publication
  (let [token-count (atom 2)
        route-policy (atom {:durable? false})
        checkpoint-args (atom nil)
        decoder {:pool ::pool :decode-state {:maxpos 8}}
        handlers
        (paged/handlers
         ::cache decoder
         {:eos-ids #{7}
          :checkpoint-policy {:expected-reuses 2.0
                              :checkpoint-ms 100.0
                              :saved-ms-per-reuse 80.0}})]
    (with-redefs [paged-decoder/prime-prompt! (fn [& _] nil)
                  paged-decoder/step!
                  (fn [& _]
                    (swap! token-count inc)
                    7)
                  page-pool/route
                  (fn [& _]
                    {:token-count @token-count :cache/policy @route-policy})
                  page-pool/route-bytes (fn [& _] 128)
                  page-pool/touch-route!
                  (fn [_ _ observations]
                    (swap! route-policy merge observations)
                    {:token-count @token-count :cache/policy @route-policy})
                  manager/stats (fn [& _] {:capture-queue-depth 0})
                  manager/checkpoint-paged-chunks-async!
                  (fn [& args]
                    (reset! checkpoint-args args)
                    {:accepted? true
                     :captured (CompletableFuture/completedFuture ::captured)
                     :published (CompletableFuture/completedFuture ::published)})]
      (let [result ((:worker/decode handlers) (effect :worker/decode))]
        (is (= [7] (:tokens result)))
        (is (true? (get-in result [:cache-checkpoint :accepted?])))
        (is (= [::cache ::pool :continuation-a "fixture-v1" [1 2 3 7]]
               @checkpoint-args))
        (is (true? (:durable? @route-policy)))
        (is (= :positive-expected-value
               (get-in @route-policy [:checkpoint/decision :reason])))))))

(deftest batched-handlers-delegate-device-work-to-one-runtime
  (let [decoder {:pool ::pool :decode-state {:maxpos 8}}
        runtime {:decoder decoder}
        calls (atom [])
        handlers (paged/batched-handlers runtime ::cache decoder)]
    (with-redefs [paged-runtime/run-operation!
                  (fn [_ _ operation]
                    (swap! calls conj :operation)
                    (operation))
                  paged-runtime/run-background-operation!
                  (fn [_ _ operation]
                    (swap! calls conj :background)
                    (operation (constantly false)))
                  manager/restore-paged-prefix-overlapped!
                  (fn [& _] {:cached-token-count 2})
                  paged-runtime/prefill!
                  (fn [_ value]
                    (swap! calls conj [:prefill (:assignment/id value)])
                    {:ok? true})
                  paged-runtime/decode!
                  (fn [_ value]
                    (swap! calls conj [:decode (:assignment/id value)])
                    {:ok? true :tokens [7 8]})
                  page-pool/route (fn [& _] {:token-count 4})
                  page-pool/route-bytes (fn [& _] 128)
                  page-pool/touch-route! (fn [& _] nil)]
      (is (= {:ok? true :cached-token-count 2}
             ((:worker/restore-prefix handlers)
              (effect :worker/restore-prefix))))
      (is (= {:ok? true}
             ((:worker/prefill-suffix handlers)
              (effect :worker/prefill-suffix))))
      (is (= {:ok? true :tokens [7 8]}
             ((:worker/decode handlers) (effect :worker/decode))))
      (is (= [:background
              :operation
              [:prefill [:request-a 1]]
              :operation
              [:decode [:request-a 1]]]
             @calls)))))

(deftest batched-prefill-attaches-a-cache-miss-route-to-its-reservation
  (let [decoder {:pool ::pool :decode-state {:maxpos 8}}
        runtime {:decoder decoder}
        allocated (atom nil)
        resident (atom nil)
        handlers (paged/batched-handlers runtime ::cache decoder)]
    (with-redefs [paged-runtime/run-operation! (fn [_ _ operation] (operation))
                  paged-runtime/prefill! (fn [& _] {:ok? true})
                  page-pool/route (fn [& _] @resident)
                  page-pool/allocate-route!
                  (fn [_ id token-count opts]
                    (let [route {:continuation-id id :token-count token-count}]
                      (reset! allocated [id token-count opts])
                      (reset! resident route)
                      route))]
      (is (= {:ok? true}
             ((:worker/prefill-suffix handlers)
              (effect :worker/prefill-suffix))))
      (is (= [:continuation-a 0
              {:policy {:durable? false}
               :capacity-reservation ::capacity}]
             @allocated)))))
