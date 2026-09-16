(ns pretrained.chat-test
  (:require [clojure.data.json :as json]
            [clojure.test :refer [deftest is]]
            [pretrained.chat :as chat])
  (:import (java.nio.file Files)
           (java.nio.file.attribute FileAttribute)))

(defn- fixture-dir
  [{:keys [chat-template eos-token eos-token-id added vocab]}]
  (let [dir (str (Files/createTempDirectory
                  "pretrained-chat-" (make-array FileAttribute 0)))]
    (spit (str dir "/tokenizer_config.json")
          (json/write-str (cond-> {}
                            chat-template (assoc "chat_template" chat-template)
                            eos-token (assoc "eos_token" eos-token))))
    (when eos-token-id
      (spit (str dir "/generation_config.json")
            (json/write-str {"eos_token_id" eos-token-id})))
    (spit (str dir "/tokenizer.json")
          (json/write-str {"added_tokens" (or added [])
                           "model" {"vocab" (or vocab {})}}))
    dir))

(def ^:private conversation
  [{:role "system" :content "Be brief."}
   {:role "user" :content "Hi"}
   {:role "assistant" :content "Hello."}
   {:role "user" :content "Capital of France?"}])

(deftest gemma-template-folds-system-into-the-first-turn
  (let [dir (fixture-dir {:chat-template "{{ bos_token }}<start_of_turn>"})
        render (chat/render-fn dir)]
    (is (= :gemma (chat/template-family dir)))
    (is (= (str "<start_of_turn>user\nBe brief.\n\nHi<end_of_turn>\n"
                "<start_of_turn>model\nHello.<end_of_turn>\n"
                "<start_of_turn>user\nCapital of France?<end_of_turn>\n"
                "<start_of_turn>model\n")
           (render conversation)))))

(deftest chatml-template-keeps-system-as-its-own-turn
  (let [dir (fixture-dir {:chat-template "<|im_start|>{{ role }}"})
        render (chat/render-fn dir)]
    (is (= :chatml (chat/template-family dir)))
    (is (= (str "<|im_start|>system\nBe brief.<|im_end|>\n"
                "<|im_start|>user\nHi<|im_end|>\n"
                "<|im_start|>assistant\nHello.<|im_end|>\n"
                "<|im_start|>user\nCapital of France?<|im_end|>\n"
                "<|im_start|>assistant\n")
           (render conversation)))
    (is (= "<|im_start|>system\nx<|im_end|>\n<|im_start|>assistant\n"
           (render [{:role "developer" :content "x"}]))
        "developer messages render as system")))

(deftest unknown-template-joins-contents
  (let [dir (fixture-dir {})]
    (is (nil? (chat/template-family dir)))
    (is (= "Hi\nHello." ((chat/render-fn dir) (rest (butlast conversation)))))))

(deftest eos-ids-combine-generation-config-and-named-token
  (is (= #{1 106}
         (chat/eos-ids (fixture-dir {:eos-token-id [1 106]
                                     :eos-token "<end_of_turn>"
                                     :added [{"content" "<end_of_turn>"
                                              "id" 106}]}))))
  (is (= #{2}
         (chat/eos-ids (fixture-dir {:eos-token "<|im_end|>"
                                     :vocab {"<|im_end|>" 2}})))
      "a checkpoint without generation_config resolves the named eos token")
  (is (= #{7} (chat/eos-ids (fixture-dir {:eos-token-id 7}))))
  (is (= #{} (chat/eos-ids (fixture-dir {})))))
