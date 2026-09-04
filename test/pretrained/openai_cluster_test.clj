(ns pretrained.openai-cluster-test
  (:require [clojure.data.json :as json]
            [clojure.test :refer [deftest is]]
            [org.httpkit.client :as client]
            [pretrained.continuation.controller.kabel :as kabel]
            [pretrained.openai.cluster :as cluster])
  (:import (java.io Closeable)))

(defrecord TestRouter [deliver! closed?]
  Closeable
  (close [_] (reset! closed? true)))

(deftest http-ingress-submits-to-and-receives-from-the-router
  (let [submitted (atom [])
        cancelled (atom [])
        opened (atom nil)]
    (with-redefs [kabel/open-router-endpoint
                  (fn [_ opts]
                    (let [router (->TestRouter (:deliver! opts) (atom false))]
                      (reset! opened router)
                      router))
                  kabel/submit!
                  (fn [router request]
                    (swap! submitted conj request)
                    ((:deliver! router)
                     {:request/id (:request/id request)
                      :response/type :completed
                      :response/value {:status :completed
                                       :tokens [65 66]
                                       :cached-token-count 1}}))
                  kabel/cancel-request!
                  (fn [_ request-id] (swap! cancelled conj request-id))]
      (let [server
            (cluster/open-server
             ::database
             {:models {"gemma-test" "gemma/fingerprint-v1"}
              :tokenize-chat (constantly [1 2])
              :decode-token #(str (char %))
              :decode-tokens #(apply str (map char %))
              :server-options {:port 0}})
            url (str "http://127.0.0.1:" (cluster/local-port server)
                     "/v1/chat/completions")]
        (try
          (let [response
                @(client/post
                  url
                  {:headers {"content-type" "application/json"}
                   :body (json/write-str
                          {:model "gemma-test"
                           :messages [{:role "user" :content "hello"}]})})
                body (json/read-str (:body response) :key-fn keyword)]
            (is (= 200 (:status response)))
            (is (= "AB" (get-in body [:choices 0 :message :content])))
            (is (= 1 (get-in body
                             [:usage :prompt_tokens_details :cached_tokens])))
            (is (= "gemma/fingerprint-v1"
                   (:request/model-fingerprint (first @submitted))))
            (is (empty? @cancelled)))
          (finally
            (.close ^Closeable server)))
        (is (true? @(:closed? @opened)))))))

(deftest failed-http-open-closes-the-router
  (let [opened (atom nil)]
    (with-redefs [kabel/open-router-endpoint
                  (fn [_ opts]
                    (let [router (->TestRouter (:deliver! opts) (atom false))]
                      (reset! opened router)
                      router))]
      (is (thrown? clojure.lang.ExceptionInfo
                   (cluster/open-server
                    ::database
                    {:models {}
                     :tokenize-chat identity
                     :decode-token str
                     :decode-tokens str})))
      (is (true? @(:closed? @opened))))))
