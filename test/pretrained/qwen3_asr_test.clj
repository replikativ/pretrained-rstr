(ns pretrained.qwen3-asr-test
  (:require [clojure.test :refer [deftest is]]
            [pretrained.asr.qwen3-asr :as qwen3-asr]
            [pretrained.decoder-gpu :as decoder-gpu]
            [raster.gpu.core :as gpu]))

(deftest gpu-rollout-advances-through-the-resident-decoder-api
  (let [steps (atom [])
        model {:d-model 2
               :tokenizer {:tok ::tokenizer
                           :decode (fn [_ token-ids] (pr-str token-ids))}}
        decode-state {:sess ::session :maxpos 8}
        prep-prompt-var (ns-resolve 'pretrained.asr.qwen3-asr 'prep-prompt)]
    (with-redefs-fn
      {prep-prompt-var
       (fn [_ _ _]
         {:prompt [10] :audio-rows (float-array 0)})
       #'decoder-gpu/decode-token!
       (fn [_ token position]
         (swap! steps conj [:prime token position]))
       #'decoder-gpu/resident-step!
       (fn [_ position]
         (swap! steps conj [:step position])
         (if (= position 1) 21 151645))
       #'gpu/download
       (fn [_ buffer]
         (is (= :tokbuf buffer))
         (int-array [20]))
       #'gpu/replay!
       (fn [& _]
         (throw (ex-info "raw graph replay is not part of the decoder API" {})))}
      (fn []
        (is (= "[20 21]"
               (qwen3-asr/transcribe-gpu model (float-array 0)
                                          {:dstate decode-state :max-new 3})))
        (is (= [[:prime 10 0] [:step 1] [:step 2]] @steps))))))
