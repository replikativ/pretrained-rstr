(ns pretrained.openai-server-test
  (:require [clojure.data.json :as json]
            [clojure.string :as str]
            [clojure.test :refer [deftest is]]
            [org.httpkit.client :as client]
            [pretrained.openai.server :as server])
  (:import (java.net Socket)
           (java.nio.charset StandardCharsets)))

(defn- request!
  [method url body]
  @(client/request
    (cond-> {:method method
             :url url
             :timeout 3000
             :headers {"content-type" "application/json"}}
      body (assoc :body (json/write-str body)))))

(defn- await!
  [timeout-ms predicate]
  (let [deadline (+ (System/nanoTime) (* 1000000 timeout-ms))]
    (loop []
      (if-let [value (predicate)]
        value
        (if (< (System/nanoTime) deadline)
          (do (Thread/sleep 5) (recur))
          (throw (ex-info "Timed out waiting for OpenAI server state"
                          {:timeout-ms timeout-ms})))))))

(defn- fixture-server
  [deliveries cancellations]
  (let [holder (atom nil)
        instance
        (server/open-server
         {:models {"gemma-test" "gemma/fingerprint-v1"}
          :tokenize-chat (fn [messages] (mapv (comp count :content) messages))
          :decode-token #(str "<" % ">")
          :decode-tokens #(apply str (map char %))
          :submit!
          (fn [request]
            (swap! deliveries conj request)
            (let [id (:request/id request)]
              (future
                (server/deliver! @holder
                                 {:request/id id :response/type :delta
                                  :response/token 65 :response/token-index 0})
                (server/deliver! @holder
                                 {:request/id id :response/type :delta
                                  :response/token 66 :response/token-index 1})
                (server/deliver! @holder
                                 {:request/id id :response/type :completed
                                  :response/value
                                  {:status :completed :tokens [65 66]
                                   :cached-token-count 1}}))))
          :cancel! #(swap! cancellations conj %)
          :server-options {:port 0}})]
    (reset! holder instance)
    instance))

(deftest model-list-and-non-streamed-chat-completion
  (let [deliveries (atom []) cancellations (atom [])
        instance (fixture-server deliveries cancellations)
        base (str "http://127.0.0.1:" (server/local-port instance))]
    (try
      (let [models (request! :get (str base "/v1/models") nil)
            completion
            (request! :post (str base "/v1/chat/completions")
                      {:model "gemma-test"
                       :messages [{:role "user" :content "hello"}]
                       :max_completion_tokens 4})
            model-body (json/read-str (:body models) :key-fn keyword)
            completion-body (json/read-str (:body completion) :key-fn keyword)]
        (is (= 200 (:status models)))
        (is (= "gemma-test" (get-in model-body [:data 0 :id])))
        (is (= 200 (:status completion)))
        (is (= "AB" (get-in completion-body [:choices 0 :message :content])))
        (is (= 2 (get-in completion-body [:usage :completion_tokens])))
        (is (= 1 (get-in completion-body
                         [:usage :prompt_tokens_details :cached_tokens])))
        (is (= "gemma/fingerprint-v1"
               (:request/model-fingerprint (first @deliveries)))))
      (finally
        (.close instance)))))

(deftest streamed-chat-completion-is-valid-sse
  (let [deliveries (atom []) cancellations (atom [])
        instance (fixture-server deliveries cancellations)
        base (str "http://127.0.0.1:" (server/local-port instance))]
    (try
      (let [response
            (request! :post (str base "/v1/chat/completions")
                      {:model "gemma-test"
                       :messages [{:role "user" :content "hello"}]
                       :max_completion_tokens 2
                       :stream true
                       :stream_options {:include_usage true}})
            body (:body response)]
        (is (= 200 (:status response)))
        (is (str/starts-with?
             (get-in response [:headers :content-type]) "text/event-stream"))
        (is (str/includes? body "\"content\":\"<65>\""))
        (is (str/includes? body "\"content\":\"<66>\""))
        (is (str/includes? body "\"completion_tokens\":2"))
        (is (str/includes? body "\"cached_tokens\":1"))
        (is (str/ends-with? body "data: [DONE]\n\n")))
      (finally
        (.close instance)))))

(deftest invalid-model-is-an-openai-error
  (let [deliveries (atom []) cancellations (atom [])
        instance (fixture-server deliveries cancellations)
        base (str "http://127.0.0.1:" (server/local-port instance))]
    (try
      (let [response
            (request! :post (str base "/v1/chat/completions")
                      {:model "missing"
                       :messages [{:role "user" :content "hello"}]})
            body (json/read-str (:body response) :key-fn keyword)]
        (is (= 400 (:status response)))
        (is (= "invalid_request_error" (get-in body [:error :type])))
        (is (empty? @deliveries)))
      (finally
        (.close instance)))))

(deftest client-disconnect-cancels-in-flight-generation
  (let [submitted (atom []) cancellations (atom [])
        instance
        (server/open-server
         {:models {"gemma-test" "gemma/fingerprint-v1"}
          :tokenize-chat (constantly [1 2])
          :decode-token str
          :decode-tokens #(apply str %)
          :submit! #(swap! submitted conj %)
          :cancel! #(swap! cancellations conj %)
          :server-options {:port 0}})
        body (json/write-str
              {:model "gemma-test"
               :messages [{:role "user" :content "hold"}]
               :stream true})
        request (str "POST /v1/chat/completions HTTP/1.1\r\n"
                     "Host: 127.0.0.1\r\n"
                     "Content-Type: application/json\r\n"
                     "Content-Length: " (count (.getBytes body StandardCharsets/UTF_8))
                     "\r\n\r\n" body)
        socket (Socket. "127.0.0.1" (int (server/local-port instance)))]
    (try
      (let [out (.getOutputStream socket)]
        (.write out (.getBytes request StandardCharsets/UTF_8))
        (.flush out))
      (let [generation (await! 1000 #(first @submitted))]
        (.close socket)
        (is (= (:request/id generation)
               (await! 1000 #(first @cancellations)))))
      (finally
        (try (.close socket) (catch Throwable _))
        (.close instance)))))
