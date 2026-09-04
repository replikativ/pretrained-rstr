(ns pretrained.continuation-kabel-demo
  "Model-free live Kabel demonstration of continuation request routing."
  (:require [clojure.data.json :as json]
            [clojure.string :as str]
            [datahike.api :as d]
            [org.httpkit.client :as http-client]
            [kabel.http-kit :as http-kit]
            [kabel.peer :as peer]
            [pretrained.attention-state :as attention-state]
            [pretrained.continuation.catalog :as catalog]
            [pretrained.continuation.controller.kabel :as controller-kabel]
            [pretrained.continuation.page-pool :as page-pool]
            [pretrained.openai.cluster :as openai-cluster]
            [superv.async :refer [<?? S]])
  (:import (java.io Closeable)
           (java.net ServerSocket)))

(def ^:private model-fingerprint "kabel-demo-model-v1")

(defn- free-port
  []
  (with-open [socket (ServerSocket. 0)]
    (.getLocalPort socket)))

(defn- device-pool
  [worker-id]
  (page-pool/->DevicePagePool
   worker-id (attention-state/layout {:n-layers 1 :n-kv 1 :head-dim 2})
   16 32 :half
   {[:key 0] [worker-id :key], [:value 0] [worker-id :value]}
   (atom {:free (apply sorted-set (range 32))
          :refcounts {} :routes {}})))

(defn- open-worker
  [worker-id queue-ms prefill-ms tokens]
  (controller-kabel/open-worker-endpoint
   (device-pool worker-id)
   {:worker/id worker-id
    :worker/epoch 0
    :worker/models #{model-fingerprint}
    :worker/free-pages 0
    :worker/evictable-pages 0}
   {:handlers {:worker/restore-prefix (fn [_] nil)
               :worker/prefill-suffix (fn [_] nil)
               :worker/decode
               (fn [effect]
                 (doseq [[index token] (map-indexed vector tokens)]
                   ((:worker/token! effect) token index))
                 {:tokens tokens})}
    :measurements {:worker/node (name worker-id)
                   :worker/queue-ms queue-ms
                   :worker/max-context 4096
                   :worker/prefill-ms-per-token prefill-ms
                   :worker/first-token-ms 1
                   :worker/gpu-restore-bytes-per-ms 1000000
                   :worker/tier-throughput-bytes-per-ms {}
                   :worker/object-store? false
                   :worker/transfer-capabilities
                   {:backend :simulated
                    :live-overlap-eligible? true}}
    :heartbeat-ms 100}))

(defn- await-value
  [timeout-ms operation]
  (let [deadline (+ (System/nanoTime) (* 1000000 timeout-ms))]
    (loop []
      (if-let [value (operation)]
        value
        (if (< (System/nanoTime) deadline)
          (do (Thread/sleep 10) (recur))
          (throw (ex-info "Timed out waiting for live Kabel demo"
                          {:timeout-ms timeout-ms})))))))

(defn run-live-simulation
  "Route one request between two workers over actual local Kabel WebSockets.

  The faster observed worker wins. The returned summary exposes the selected
  worker, delivered tokens, assignment phase, and live observation count. No
  model weights, tensor payloads, or external services are required."
  []
  (let [config {:store {:backend :memory :id (random-uuid)}
                :schema-flexibility :write
                :keep-history? false
                :value-caps :default}
        connection (catalog/ensure-database! config)
        delivered (atom [])
        router-endpoint (controller-kabel/open-router-endpoint
                         connection
                         {:deliver! #(swap! delivered conj %)
                          :heartbeat-timeout-ms 1000
                          :offer-timeout-ms 250
                          :chunk-size 16})
        server-id (random-uuid)
        url (str "ws://localhost:" (free-port))
        server-peer (peer/server-peer
                     S (http-kit/create-http-kit-handler! S url server-id)
                     server-id
                     (controller-kabel/router-middleware router-endpoint))
        fast-endpoint (open-worker :fast-gpu 0 0.5 [101 102])
        slow-endpoint (open-worker :busy-gpu 30 2 [201 202])
        fast-peer (peer/client-peer
                   S (random-uuid)
                   (controller-kabel/worker-middleware fast-endpoint))
        slow-peer (peer/client-peer
                   S (random-uuid)
                   (controller-kabel/worker-middleware slow-endpoint))]
    (try
      (<?? S (peer/start server-peer))
      (<?? S (peer/connect S fast-peer url))
      (<?? S (peer/connect S slow-peer url))
      (await-value 3000
                   #(when (= 2
                             (count
                              (controller-kabel/observations router-endpoint)))
                      true))
      (controller-kabel/submit!
       router-endpoint
       {:request/id :kabel-demo-request
        :request/model-fingerprint model-fingerprint
        :request/tokens (vec (range 65))
        :request/max-new-tokens 8})
      (let [response
            (await-value
             3000
             #(some (fn [delivery]
                      (when (= :completed (:response/type delivery)) delivery))
                    @delivered))
            assignment (get-in
                        (controller-kabel/router-state router-endpoint)
                        [:router/requests :kabel-demo-request])]
        {:selected-worker
         (get-in assignment [:assignment/candidate :candidate/worker-id])
         :tokens (get-in response [:response/value :tokens])
         :phase (:assignment/phase assignment)
         :observed-workers
         (count (controller-kabel/observations router-endpoint))})
      (finally
        (.close ^Closeable fast-endpoint)
        (.close ^Closeable slow-endpoint)
        (.close ^Closeable router-endpoint)
        (<?? S (peer/stop server-peer))
        (d/release connection)
        (d/delete-database config)))))

(defn- parse-sse-values
  [body]
  (->> (str/split-lines body)
       (keep (fn [line]
               (when (and (str/starts-with? line "data: ")
                          (not= "data: [DONE]" line))
                 (json/read-str (subs line 6) :key-fn keyword))))))

(defn run-openai-live-simulation
  "Serve one streamed OpenAI request through two actual Kabel worker sockets.

  This is a model-free executable topology test: HTTP ingress, Datahike-backed
  candidate planning, WebSocket offers, ordered token deltas, terminal usage,
  and shutdown all use the production adapters. Tensor payloads and a real
  decoder can replace the fixture handlers without changing the gateway."
  []
  (let [config {:store {:backend :memory :id (random-uuid)}
                :schema-flexibility :write
                :keep-history? false
                :value-caps :default}
        connection (catalog/ensure-database! config)
        gateway (openai-cluster/open-server
                 connection
                 {:models {"gemma-demo" model-fingerprint}
                  :tokenize-chat (constantly (vec (range 65)))
                  :decode-token #(str "<" % ">")
                  :decode-tokens #(apply str (map (fn [token]
                                                    (str "<" token ">")) %))
                  :server-options {:port 0}
                  :router-options {:heartbeat-timeout-ms 1000
                                   :offer-timeout-ms 250
                                   :chunk-size 16}})
        server-id (random-uuid)
        ws-url (str "ws://localhost:" (free-port))
        server-peer (peer/server-peer
                     S (http-kit/create-http-kit-handler! S ws-url server-id)
                     server-id
                     (openai-cluster/router-middleware gateway))
        fast-endpoint (open-worker :fast-gpu 0 0.5 [101 102])
        slow-endpoint (open-worker :busy-gpu 30 2 [201 202])
        fast-peer (peer/client-peer
                   S (random-uuid)
                   (controller-kabel/worker-middleware fast-endpoint))
        slow-peer (peer/client-peer
                   S (random-uuid)
                   (controller-kabel/worker-middleware slow-endpoint))]
    (try
      (<?? S (peer/start server-peer))
      (<?? S (peer/connect S fast-peer ws-url))
      (<?? S (peer/connect S slow-peer ws-url))
      (await-value 3000
                   #(when (= 2 (count (openai-cluster/observations gateway)))
                      true))
      (let [url (str "http://127.0.0.1:" (openai-cluster/local-port gateway)
                     "/v1/chat/completions")
            response
            @(http-client/post
              url
              {:timeout 5000
               :headers {"content-type" "application/json"}
               :body (json/write-str
                      {:model "gemma-demo"
                       :messages [{:role "user" :content "route this"}]
                       :max_completion_tokens 8
                       :stream true
                       :stream_options {:include_usage true}})})
            values (vec (parse-sse-values (:body response)))
            request-id (:id (first values))
            assignment (get-in (openai-cluster/router-state gateway)
                               [:router/requests request-id])
            usage (:usage (some #(when (:usage %) %) values))]
        {:http-status (:status response)
         :content-type (get-in response [:headers :content-type])
         :selected-worker
         (get-in assignment [:assignment/candidate :candidate/worker-id])
         :phase (:assignment/phase assignment)
         :text (apply str (keep #(get-in % [:choices 0 :delta :content]) values))
         :cached-token-count (get-in usage
                                     [:prompt_tokens_details :cached_tokens])
         :observed-workers (count (openai-cluster/observations gateway))})
      (finally
        (.close ^Closeable fast-endpoint)
        (.close ^Closeable slow-endpoint)
        (.close ^Closeable gateway)
        (<?? S (peer/stop server-peer))
        (d/release connection)
        (d/delete-database config)))))
