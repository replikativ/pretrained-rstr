(ns pretrained.continuation.controller.paged-test
  (:require [clojure.test :refer [deftest is]]
            [pretrained.continuation.controller.paged :as paged]
            [pretrained.continuation.manager :as manager]
            [pretrained.continuation.page-pool :as page-pool]
            [pretrained.continuation.paged-decoder :as paged-decoder]))

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
