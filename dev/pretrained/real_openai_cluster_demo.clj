(ns pretrained.real-openai-cluster-demo
  "Resource-gated real-model OpenAI/Kabel/paged-KV cluster smoke.

  Both advertised workers own an independent Raster decode session and page
  pool. The model and its Q4 host packing are shared to bound host memory."
  (:refer-clojure :exclude [run!])
  (:require [clojure.data.json :as json]
            [clojure.string :as str]
            [datahike.api :as d]
            [kabel.http-kit :as http-kit]
            [kabel.peer :as peer]
            [org.httpkit.client :as http-client]
            [pretrained.continuation.catalog :as catalog]
            [pretrained.continuation.controller.kabel :as controller-kabel]
            [pretrained.continuation.controller.paged :as paged-controller]
            [pretrained.continuation.manager :as manager]
            [pretrained.continuation.page-pool :as page-pool]
            [pretrained.continuation.paged-decoder :as paged-decoder]
            [pretrained.decoder-gpu :as decoder-gpu]
            [pretrained.loader :as loader]
            [pretrained.model-identity :as model-identity]
            [pretrained.openai.cluster :as openai-cluster]
            [raster.gpu.core :as gpu]
            [superv.async :refer [<?? S]])
  (:import (java.io Closeable)
           (java.net ServerSocket)
           (java.nio.file Files Path)
           (java.util Comparator)))

(def ^:private gib (* 1024 1024 1024))

(defn- read-lines
  [path]
  ;; GraalVM's BufferedInputStream calls FileInputStream.available(), which
  ;; returns EINVAL for procfs. The NIO channel reader does not use available().
  (vec (Files/readAllLines (Path/of path (make-array String 0)))))

(defn resource-snapshot
  "Return the Linux resource facts used to guard the two-worker smoke."
  []
  (let [meminfo
        (into {}
              (keep (fn [line]
                      (when-let [[_ field kib]
                                 (re-matches #"([^:]+):\s+(\d+)\s+kB" line)]
                        [(keyword field) (* 1024 (Long/parseLong kib))])))
              (read-lines "/proc/meminfo"))
        load-one (Double/parseDouble
                  (first (str/split (first (read-lines "/proc/loadavg")) #"\s+")))]
    {:available-bytes (:MemAvailable meminfo)
     :swap-total-bytes (:SwapTotal meminfo)
     :swap-free-bytes (:SwapFree meminfo)
     :load-one load-one
     :processors (.availableProcessors (Runtime/getRuntime))}))

(defn resource-admission
  "Explain whether `snapshot` is safe for a two-worker compile."
  ([snapshot] (resource-admission snapshot {}))
  ([{:keys [available-bytes swap-total-bytes swap-free-bytes load-one processors]
     :as snapshot}
    {:keys [minimum-available-bytes minimum-swap-free-bytes
            maximum-load-per-processor]
     :or {minimum-available-bytes (* 14 gib)
          minimum-swap-free-bytes (* 256 1024 1024)
          maximum-load-per-processor 1.25}}]
   (let [load-per-processor (/ (double load-one) (max 1 (long processors)))
         reasons
         (cond-> []
           (< (long available-bytes) (long minimum-available-bytes))
           (conj :insufficient-available-memory)
           (and (pos? (long swap-total-bytes))
                (< (long swap-free-bytes) (long minimum-swap-free-bytes)))
           (conj :insufficient-free-swap)
           (> load-per-processor (double maximum-load-per-processor))
           (conj :host-load-too-high))]
     {:admitted? (empty? reasons)
      :reasons reasons
      :load-per-processor load-per-processor
      :snapshot snapshot})))

(defn preflight
  "Return current resource admission without loading weights or opening a GPU."
  ([] (preflight {}))
  ([thresholds] (resource-admission (resource-snapshot) thresholds)))

(defn- require-admission!
  [opts]
  (let [admission (preflight (:resource-thresholds opts))]
    (when (and (not (:force? opts)) (not (:admitted? admission)))
      (throw (ex-info
              "Refusing two-worker compile under current host pressure; retry later or pass :force? true"
              admission)))
    admission))

(defn- free-port
  []
  (with-open [socket (ServerSocket. 0)] (.getLocalPort socket)))

(defn- delete-tree!
  [^Path directory]
  (when (Files/exists directory (make-array java.nio.file.LinkOption 0))
    (with-open [paths (Files/walk directory
                                  (make-array java.nio.file.FileVisitOption 0))]
      (doseq [path (iterator-seq
                    (.iterator (.sorted paths (Comparator/reverseOrder))))]
        (Files/deleteIfExists path)))))

(defrecord ModelWorker [endpoint cache decoder decode-state]
  Closeable
  (close [_]
    (try
      (when endpoint (.close ^Closeable endpoint))
      (finally
        (try
          (when cache (.close ^Closeable cache))
          (finally
            (try
              (when decoder
                (try
                  (paged-decoder/close! decoder)
                  (finally
                    (page-pool/close-transfer-engines! (:pool decoder)))))
              (finally
                (when decode-state
                  (gpu/close-session! (:sess decode-state)))))))))))

(defn- open-model-worker!
  [model quantized-weights fingerprint connection cache-directory worker-id opts]
  (let [max-position (long (:max-position opts 64))
        page-size (long (:page-size opts 16))
        physical-pages (long (:physical-pages opts 8))
        decode-state (volatile! nil)
        decoder (volatile! nil)
        cache (volatile! nil)
        endpoint (volatile! nil)]
    (try
      (vreset! decode-state
               (decoder-gpu/bind-decode!
                model :qw quantized-weights :maxpos max-position
                :cache-mode :paged :batch-size 1
                :device-id (:device-id opts :ze:0)))
      (vreset! decoder
               (paged-decoder/open!
                @decode-state :page-size page-size
                :physical-pages physical-pages :key-prefix (name worker-id)))
      (vreset! cache
               (manager/open-manager
                nil cache-directory
                {:connection connection
                 :chunk-size (long (:chunk-size opts 4))
                 :max-pending-captures 1
                 :max-pending-publications 1}))
      (vreset! endpoint
               (controller-kabel/open-worker-endpoint
                (:pool @decoder)
                {:worker/id worker-id :worker/epoch 0
                 :worker/models #{fingerprint}
                 :worker/free-pages physical-pages :worker/evictable-pages 0}
                {:handlers (paged-controller/handlers
                            @cache @decoder {:chunk-size (:chunk-size opts 4)})
                 :measurements
                 {:worker/node (name worker-id)
                  :worker/queue-ms (double (:queue-ms opts 0.0))
                  :worker/max-context max-position
                  :worker/prefill-ms-per-token
                  (double (:prefill-ms-per-token opts 10.0))
                  :worker/first-token-ms (double (:first-token-ms opts 10.0))
                  :worker/gpu-restore-bytes-per-ms 1000000.0
                  :worker/tier-throughput-bytes-per-ms {}
                  :worker/object-store? false}
                 :heartbeat-ms (long (:heartbeat-ms opts 250))}))
      (->ModelWorker @endpoint @cache @decoder @decode-state)
      (catch Throwable error
        (try
          (.close (->ModelWorker @endpoint @cache @decoder @decode-state))
          (catch Throwable _))
        (throw error)))))

(defn- await-value
  [timeout-ms operation]
  (let [deadline (+ (System/nanoTime) (* 1000000 (long timeout-ms)))]
    (loop []
      (if-let [value (operation)]
        value
        (if (< (System/nanoTime) deadline)
          (do (Thread/sleep 20) (recur))
          (throw (ex-info "Timed out waiting for real cluster state"
                          {:timeout-ms timeout-ms})))))))

(defn- post-completion!
  [port model-id content max-new-tokens]
  (let [response
        @(http-client/post
          (str "http://127.0.0.1:" port "/v1/chat/completions")
          {:timeout 120000
           :headers {"content-type" "application/json"}
           :body (json/write-str
                  {:model model-id
                   :messages [{:role "user" :content content}]
                   :max_completion_tokens max-new-tokens})})]
    (when-not (= 200 (:status response))
      (throw (ex-info "OpenAI completion failed" {:response response})))
    (json/read-str (:body response) :key-fn keyword)))

(defn run!
  "Run two real workers and prove resident KV reuse through the OpenAI API.

  Defaults to `~/Development/models/gemma-3-270m-it`, max position 64, and
  eight physical pages per worker. Pass `:force? true` only after independently
  checking host and integrated-GPU pressure."
  ([] (run! {}))
  ([opts]
   (let [admission (require-admission! opts)
         model-directory (or (:model-directory opts)
                             (str (System/getProperty "user.home")
                                  "/Development/models/gemma-3-270m-it"))
         model-id (:model-id opts "gemma-3-270m-it")
         config {:store {:backend :memory :id (random-uuid)}
                 :schema-flexibility :write :keep-history? false
                 :value-caps :default}
         connection (catalog/ensure-database! config)
         temp-directory (Files/createTempDirectory
                         "pretrained-real-cluster-"
                         (make-array java.nio.file.attribute.FileAttribute 0))
         gateway (volatile! nil)
         server-peer (volatile! nil)
         workers (atom [])
         client-peers (atom [])]
     (try
       (let [model (loader/from-pretrained model-directory)
             fingerprint
             (model-identity/compatibility-fingerprint
              model (cond-> {:execution-variant :gpu-q4k-paged}
                      (:weights-id opts) (assoc :weights-id (:weights-id opts))))
             quantized-weights (decoder-gpu/gpu-quantize model)
             {:keys [tok encode decode]} (:tokenizer model)
             tokenize-chat
             (fn [messages]
               (vec (encode tok (str/join "\n" (map :content messages)))))
             _ (vreset! gateway
                        (openai-cluster/open-server
                         connection
                         {:models {model-id fingerprint}
                          :tokenize-chat tokenize-chat
                          :decode-token #(decode tok [%])
                          :decode-tokens #(decode tok %)
                          :server-options {:port 0}
                          :router-options {:heartbeat-timeout-ms 3000
                                           :offer-timeout-ms 1000
                                           :chunk-size (:chunk-size opts 4)}}))
             server-id (random-uuid)
             ws-url (str "ws://localhost:" (free-port))
             _ (vreset! server-peer
                        (peer/server-peer
                         S (http-kit/create-http-kit-handler! S ws-url server-id)
                         server-id (openai-cluster/router-middleware @gateway)))
             _ (<?? S (peer/start @server-peer))]
         (doseq [[worker-id queue-ms] [[:worker-a 0.0] [:worker-b 1.0]]]
           (let [worker
                 (open-model-worker!
                  model quantized-weights fingerprint connection
                  (.resolve temp-directory (name worker-id)) worker-id
                  (assoc opts :queue-ms queue-ms))
                 client-peer
                 (peer/client-peer
                  S (random-uuid)
                  (controller-kabel/worker-middleware (:endpoint worker)))]
             (swap! workers conj worker)
             (<?? S (peer/connect S client-peer ws-url))
             (swap! client-peers conj client-peer)))
         (await-value 10000
                      #(when (= 2 (count (openai-cluster/observations @gateway)))
                         true))
         (let [prompt (:prompt opts "The capital of France is")
               max-new (:max-new-tokens opts 2)
               first-response
               (post-completion! (openai-cluster/local-port @gateway)
                                 model-id prompt max-new)
               first-id (:id first-response)
               first-assignment
               (get-in (openai-cluster/router-state @gateway)
                       [:router/requests first-id])
               output (get-in first-assignment [:assignment/result :tokens])
               prompt-tokens
               (get-in first-assignment [:assignment/request :request/tokens])
               continued-tokens (into (vec prompt-tokens) output)
               continued-text (decode tok continued-tokens)
               roundtrip-tokens (vec (encode tok continued-text))
               _ (when-not (= continued-tokens roundtrip-tokens)
                   (throw (ex-info
                           "Tokenizer did not preserve exact continuation history"
                           {:expected continued-tokens :actual roundtrip-tokens})))
               _ (await-value
                  5000
                  #(when (some (comp seq :worker/gpu-prefixes)
                               (openai-cluster/observations @gateway))
                     true))
               second-response
               (post-completion! (openai-cluster/local-port @gateway)
                                 model-id continued-text max-new)
               second-id (:id second-response)
               second-assignment
               (get-in (openai-cluster/router-state @gateway)
                       [:router/requests second-id])
               cached-tokens
               (get-in second-response
                       [:usage :prompt_tokens_details :cached_tokens])]
           (when-not (pos? (long cached-tokens))
             (throw (ex-info "Second request did not reuse resident KV"
                             {:response second-response
                              :assignment second-assignment})))
           {:resource-admission admission
            :model model-id
            :observed-workers (count (openai-cluster/observations @gateway))
            :first-worker
            (get-in first-assignment
                    [:assignment/candidate :candidate/worker-id])
            :second-worker
            (get-in second-assignment
                    [:assignment/candidate :candidate/worker-id])
            :first-text (get-in first-response [:choices 0 :message :content])
            :second-text (get-in second-response [:choices 0 :message :content])
            :cached-tokens cached-tokens
            :second-usage (:usage second-response)}))
       (finally
         (doseq [client-peer (reverse @client-peers)]
           (try (<?? S (peer/stop client-peer)) (catch Throwable _)))
         (doseq [worker (reverse @workers)]
           (try (.close ^Closeable worker) (catch Throwable _)))
         (when @gateway
           (try (.close ^Closeable @gateway) (catch Throwable _)))
         (when @server-peer
           (try (<?? S (peer/stop @server-peer)) (catch Throwable _)))
         (d/release connection)
         (d/delete-database config)
         (delete-tree! temp-directory))))))
