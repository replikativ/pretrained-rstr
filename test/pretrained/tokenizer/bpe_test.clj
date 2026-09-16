(ns pretrained.tokenizer.bpe-test
  "Validates byte-level BPE chat encoding against transformers-generated gold ids
  for SmolLM2-360M-Instruct (ChatML). Skips when the tokenizer is absent."
  (:require [clojure.test :refer [deftest is]]
            [pretrained.tokenizer.bpe :as bpe]))

(def ^:private tok-path
  (or (System/getenv "SMOLLM_TOKENIZER")
      "/home/christian-weilbach/Development/models/SmolLM2-360M-Instruct/tokenizer.json"))

(def ^:private present? (.exists (java.io.File. tok-path)))

;; gold from `AutoTokenizer.apply_chat_template(tokenize=True, add_generation_prompt=True)`.
(def ^:private chat-gold
  {"<|im_start|>user\nHi<|im_end|>\n<|im_start|>assistant\n"
   [1 4093 198 26843 2 198 1 520 9531 198]
   (str "<|im_start|>system\nBe brief.<|im_end|>\n"
        "<|im_start|>user\nHi<|im_end|>\n"
        "<|im_start|>assistant\nHello.<|im_end|>\n"
        "<|im_start|>user\nCapital of France?<|im_end|>\n"
        "<|im_start|>assistant\n")
   [1 9690 198 6077 5453 30 2 198 1 4093 198 26843 2 198 1 520 9531 198 19556
    30 2 198 1 4093 198 39832 282 4649 47 2 198 1 520 9531 198]})

(deftest smollm-special-tokens-encode-to-their-ids
  (if-not present?
    (println "  [skip] SmolLM tokenizer not at" tok-path "— set SMOLLM_TOKENIZER")
    (let [t (bpe/load-bpe-tokenizer tok-path)]
      (doseq [[s ids] chat-gold]
        (is (= ids (vec (bpe/encode t s))) (str "chat encode " (pr-str s)))))))
