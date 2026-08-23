(ns pretrained.device-selection-test
  (:require [clojure.test :refer [deftest is]]
            [pretrained.asr.qwen3-asr :as qwen3-asr]
            [pretrained.decoder-gpu :as decoder-gpu]
            [pretrained.embed :as embed]
            [pretrained.lm :as lm]
            [pretrained.loader :as loader]))

(deftest rejects-an-unsupported-managed-session-before-model-work
  (is (thrown-with-msg? clojure.lang.ExceptionInfo #"Use :ze:N or :ocl:N"
                        (decoder-gpu/bind-decode! {} :device-id :cuda:0))))

(deftest task-apis-thread-the-selected-device
  (let [calls (atom [])]
    (with-redefs [loader/from-pretrained (fn [_] {})
                  decoder-gpu/quantize-q8s (fn [_] ::quantized)
                  decoder-gpu/bind-embed!
                  (fn [_ & args] (swap! calls conj [:embed (apply hash-map args)]) ::embed)
                  decoder-gpu/bind-decode!
                  (fn [_ & args] (swap! calls conj [:decode (apply hash-map args)]) ::decode)]
      (is (= ::decode (::lm/decode
                       (lm/load-lm :gemma-3-270m-it "/fixture"
                                   {:gpu? true :device-id :ocl:3 :maxpos 17}))))
      (is (= ::embed (::embed/gpu
                      (embed/load-embedder :qwen3-embedding-0.6b-gpu "/fixture"
                                           {:device-id :ocl:2 :T 19}))))
      (is (= ::decode
             (qwen3-asr/bind-gpu {} :device-id :ocl:1 :maxpos 23)))
      (is (= [[:decode {:maxpos 17 :device-id :ocl:3}]
              [:embed {:T 19 :device-id :ocl:2 :qw ::quantized}]
              [:decode {:maxpos 23 :prefill-T nil :device-id :ocl:1}]]
             @calls)))))
