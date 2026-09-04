(ns pretrained.openai-test
  (:require [clojure.data.json :as json]
            [clojure.test :refer [deftest is testing]]
            [pretrained.openai :as openai]))

(def options
  {:models {"gemma-test" "gemma/fingerprint-v1"}
   :tokenize-chat (fn [messages]
                    (mapv (comp count :content) messages))
   :request-id (constantly "chatcmpl-test")
   :created (constantly 42)})

(deftest chat-request-normalizes-at-the-http-boundary
  (let [{context :openai/context request :generation/request}
        (openai/normalize-chat-request
         {:model "gemma-test"
          :messages [{:role "system" :content "abc"}
                     {:role "user" :content "hello"}]
          :max_completion_tokens 12
          :temperature 0.5
          :top_p 0.9
          :stream true}
         options)]
    (is (= {:id "chatcmpl-test" :object "chat.completion"
            :created 42 :model "gemma-test" :stream? true
            :prompt-tokens 2 :request/max-new-tokens 12
            :include-usage? false}
           context))
    (is (= [3 5] (:request/tokens request)))
    (is (= 12 (:request/max-new-tokens request)))
    (is (= "gemma/fingerprint-v1" (:request/model-fingerprint request)))
    (is (= {:temperature 0.5 :top-p 0.9 :stop [] :seed nil}
           (:request/sampling request)))))

(deftest invalid-requests-return-openai-shaped-errors
  (doseq [body [{:model "missing" :messages [{:role "user" :content "x"}]}
                {:model "gemma-test" :messages []}
                {:model "gemma-test"
                 :messages [{:role "tool" :content "x"}]}]]
    (try
      (openai/normalize-chat-request body options)
      (is false "invalid request was accepted")
      (catch clojure.lang.ExceptionInfo error
        (is (= "invalid_request_error"
               (get-in (openai/error-object error) [:error :type])))))))

(deftest controller-deliveries-become-stream-chunks
  (let [context {:id "chatcmpl-test" :created 42 :model "gemma-test"
                 :prompt-tokens 4 :request/max-new-tokens 2
                 :include-usage? false :decode-token #(str "<" % ">")}
        delta (first (openai/stream-values
                      context {:response/type :delta :response/token 7}))
        terminal (openai/stream-values
                  context {:response/type :completed
                           :response/value {:status :completed :tokens [7 8]}})]
    (is (= "<7>" (get-in delta [:choices 0 :delta :content])))
    (is (= "length" (get-in (first terminal)
                             [:choices 0 :finish_reason])))
    (is (= :done (last terminal)))
    (is (= "[DONE]" (subs (openai/sse :done) 6 12)))
    (is (= "chat.completion.chunk"
           (:object (json/read-str
                     (subs (openai/sse delta) 6
                           (- (count (openai/sse delta)) 2))
                     :key-fn keyword))))))

(deftest non-streamed-response-has-text-and-usage
  (let [response
        (openai/completion-response
         {:id "chatcmpl-test" :created 42 :model "gemma-test"
          :prompt-tokens 3 :request/max-new-tokens 4
          :decode-tokens #(apply str (map char %))}
         {:response/type :completed
          :response/value {:status :completed :tokens [65 66]}})]
    (is (= "AB" (get-in response [:choices 0 :message :content])))
    (is (= "stop" (get-in response [:choices 0 :finish_reason])))
    (is (= {:prompt_tokens 3 :completion_tokens 2 :total_tokens 5}
           (:usage response)))))
