(ns pretrained.openai-local-test
  (:require [clojure.data.json :as json]
            [clojure.test :refer [deftest is]]
            [datahike.api :as d]
            [org.httpkit.client :as client]
            [pretrained.attention-state :as attention-state]
            [pretrained.continuation.catalog :as catalog]
            [pretrained.continuation.page-pool :as page-pool]
            [pretrained.openai.local :as local-server])
  (:import (java.io Closeable)))

(def ^:private fingerprint "fixture-model-v1")

(defn- fixture-pool
  [physical-pages]
  (page-pool/->DevicePagePool
   ::session (attention-state/layout {:n-layers 1 :n-kv 1 :head-dim 2})
   2 physical-pages :half
   {[:key 0] :pool-k0, [:value 0] :pool-v0}
   (atom {:free (apply sorted-set (range physical-pages))
          :refcounts {} :routes {}})))

(def ^:private measurements
  {:worker/node "worker-a"
   :worker/queue-ms 0.0
   :worker/max-context 32
   :worker/prefill-ms-per-token 1.0
   :worker/first-token-ms 1.0
   :worker/gpu-restore-bytes-per-ms 1000000.0
   :worker/tier-throughput-bytes-per-ms {}
   :worker/object-store? false
   :worker/transfer-capabilities {:submission :inline
                                  :independent-physical-queue? false}})

(defn- open-fixture-server
  [connection decoded]
  (local-server/open-server-with-worker
   connection
   {:pool (fixture-pool 8)
    :worker-opts {:worker/id :worker-a :worker/epoch 0
                  :worker/models #{fingerprint}
                  :worker/free-pages 8 :worker/evictable-pages 0}
    :handlers {:worker/restore-prefix (fn [_] {:ok? true :cached-token-count 0})
               :worker/prefill-suffix (fn [_] {:ok? true})
               :worker/decode
               (fn [effect]
                 (let [token! (:worker/token! effect)]
                   (swap! decoded conj (:request/tokens
                                        (:assignment/request effect)))
                   (token! 65 0)
                   (token! 66 1)
                   {:ok? true :tokens [65 66]}))}
    :measurements measurements
    :models {"fixture" fingerprint}
    :tokenize-chat (fn [messages] (mapv (comp count :content) messages))
    :decode-token #(str (char %))
    :decode-tokens #(apply str (map char %))
    :server-options {:port 0}
    :chunk-size 2}))

(defn- with-catalog
  [f]
  (let [config {:store {:backend :memory :id (random-uuid)}
                :schema-flexibility :write :keep-history? false
                :value-caps :default}
        connection (catalog/ensure-database! config)]
    (try
      (f connection)
      (finally
        (d/release connection)
        (d/delete-database config)))))

(deftest in-process-router-and-worker-serve-a-chat-completion
  (with-catalog
    (fn [connection]
      (let [decoded (atom [])
            server (open-fixture-server connection decoded)
            url (str "http://127.0.0.1:" (local-server/local-port server)
                     "/v1/chat/completions")]
        (try
          (let [response
                @(client/post
                  url {:timeout 5000
                       :headers {"content-type" "application/json"}
                       :body (json/write-str
                              {:model "fixture"
                               :messages [{:role "user" :content "hello"}]
                               :max_completion_tokens 2})})
                body (json/read-str (:body response) :key-fn keyword)]
            (is (= 200 (:status response)) (:body response))
            (is (= "AB" (get-in body [:choices 0 :message :content])))
            (is (= 1 (get-in body [:usage :prompt_tokens]))
                "one token per message from the fixture tokenizer")
            (is (= [[5]] @decoded)
                "the worker decoded the tokenized request")
            (is (= :completed
                   (get-in (local-server/router-state server)
                           [:router/requests (:id body) :assignment/phase]))
                "the router saw the fenced terminal result"))
          (let [response
                @(client/post
                  url {:timeout 5000
                       :headers {"content-type" "application/json"}
                       :body (json/write-str
                              {:model "fixture"
                               :stream true
                               :messages [{:role "user" :content "hi"}]})})]
            (is (= 200 (:status response)))
            (is (re-find #"data: \[DONE\]" (:body response))
                "streaming deliveries reach the SSE writer"))
          (finally
            (.close ^Closeable server)))
        (is (true? @(:closed? (:router server))))
        (is (true? @(:closed? (:controller server))))))))

(deftest ingress-failure-closes-router-and-controller
  (with-catalog
    (fn [connection]
      (is (thrown? clojure.lang.ExceptionInfo
                   (local-server/open-server-with-worker
                    connection
                    {:pool (fixture-pool 2)
                     :worker-opts {:worker/id :worker-a :worker/epoch 0
                                   :worker/models #{fingerprint}
                                   :worker/free-pages 2
                                   :worker/evictable-pages 0}
                     :handlers {:worker/restore-prefix (fn [_] nil)
                                :worker/prefill-suffix (fn [_] nil)
                                :worker/decode (fn [_] {:tokens []})}
                     :measurements measurements
                     ;; Empty model map is rejected by the HTTP server.
                     :models {}
                     :tokenize-chat identity
                     :decode-token str
                     :decode-tokens str
                     :server-options {:port 0}}))))))
