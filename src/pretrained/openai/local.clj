(ns pretrained.openai.local
  "Serve one local model through the OpenAI-compatible boundary in one process.

  The cluster router and the worker-local controller are joined in memory
  through the same `wire` conversions the Kabel transport uses, so offers,
  acknowledgements, token deltas, and terminal results are fenced exactly as
  they are across a socket. No Kabel peer, WebSocket, or heartbeat is involved.
  Datahike remains the catalog: in memory by default, or file-backed under
  `:cache-directory` so durable chunks and their identities survive a restart.

  Requires the Replikativ HTTP-kit distribution (`:openai-server` alias)."
  (:require [datahike.api :as d]
            [pretrained.chat :as chat]
            [pretrained.continuation.catalog :as catalog]
            [pretrained.continuation.controller.candidates :as candidates]
            [pretrained.continuation.controller.cluster :as cluster]
            [pretrained.continuation.controller.local :as local]
            [pretrained.continuation.controller.wire :as wire]
            [pretrained.continuation.model-worker :as model-worker]
            [pretrained.decoder-gpu :as decoder-gpu]
            [pretrained.lm :as lm]
            [pretrained.loader :as loader]
            [pretrained.model-identity :as model-identity]
            [pretrained.openai.server :as openai-server])
  (:import (java.io Closeable File)
           (java.nio.file Files Path)
           (java.nio.file.attribute FileAttribute)
           (java.util Comparator)))

(defn- delete-tree!
  [^Path directory]
  (when (Files/exists directory (make-array java.nio.file.LinkOption 0))
    (with-open [paths (Files/walk directory
                                  (make-array java.nio.file.FileVisitOption 0))]
      (doseq [path (iterator-seq
                    (.iterator (.sorted paths (Comparator/reverseOrder))))]
        (Files/deleteIfExists path)))))

(defrecord LocalOpenAIServer
    [http router controller worker connection datahike-config ephemeral?
     temp-directory model-id fingerprint closed?]
  Closeable
  (close [_]
    (when (compare-and-set! closed? false true)
      ;; Ingress first, so in-flight requests are cancelled while the router
      ;; and worker can still fence them; then router, controller, worker,
      ;; and finally the catalog.
      (try
        (.close ^Closeable http)
        (finally
          (try
            (.close ^Closeable router)
            (finally
              (try
                (.close ^Closeable controller)
                (finally
                  (try
                    (when worker (.close ^Closeable worker))
                    (finally
                      (when connection
                        (try (d/release connection) (catch Throwable _)))
                      (when (and ephemeral? datahike-config)
                        (try (d/delete-database datahike-config)
                             (catch Throwable _)))
                      (when temp-directory
                        (delete-tree! temp-directory)))))))))))))

(defn open-server-with-worker
  "Join an already opened worker to a router and HTTP ingress in one process.

  `connection` is the Datahike catalog connection or value used for candidate
  planning. Options require `:pool`, `:worker-opts`, `:handlers`, and
  `:measurements` (a map or zero-argument function) as produced by
  `pretrained.continuation.model-worker/open-worker!`, plus the ingress
  options `:models`, `:tokenize-chat`, `:decode-token`, and `:decode-tokens`.
  Optional `:server-options` (HTTP-kit; `:port` defaults to 8080),
  `:offer-timeout-ms` (default 100), `:chunk-size` for candidate planning
  (default 256), `:max-pending-events`, and `:on-error!`."
  [connection
   {:keys [pool worker-opts handlers measurements models tokenize-chat
           decode-token decode-tokens server-options offer-timeout-ms
           chunk-size max-pending-events on-error!]
    :or {offer-timeout-ms 100
         chunk-size 256
         on-error! (fn [error] (.printStackTrace ^Throwable error))}}]
  (when-not (and (map? worker-opts) (:worker/id worker-opts))
    (throw (ex-info "Local server requires worker options with :worker/id" {})))
  (when-not (or (map? measurements) (fn? measurements))
    (throw (ex-info "Local server requires scheduling measurements" {})))
  (let [worker-id (:worker/id worker-opts)
        router-holder (atom nil)
        http-holder (atom nil)
        guard (fn [thunk]
                (try (thunk) (catch Throwable error (on-error! error))))
        controller
        (local/open-controller
         pool worker-opts
         {:handlers handlers
          :send! (fn [effect]
                   (guard
                    #(when-let [router @router-holder]
                       (when-not @(:closed? router)
                         (when-let [event (wire/router-event
                                           (wire/effect->message effect))]
                           (cluster/handle-event! router event))))))})
        router
        (try
          (cluster/open-controller
           {:send! (fn [effect]
                     (guard
                      #(when-let [event (wire/worker-event
                                         worker-id
                                         (wire/effect->message effect))]
                         (local/handle-event! controller event))))
            :deliver! (fn [delivery]
                        (when-let [http @http-holder]
                          (openai-server/deliver! http delivery)))
            :offer-timeout-ms offer-timeout-ms})
          (catch Throwable error
            (.close ^Closeable controller)
            (throw error)))
        _ (reset! router-holder router)
        sequence (atom 0)
        observe
        (fn []
          (let [current (if (fn? measurements) (measurements) measurements)]
            (candidates/worker-observation
             (assoc (local/observation controller current)
                    :worker/sequence (swap! sequence inc)))))
        database-value
        (fn []
          (if (instance? clojure.lang.IDeref connection)
            @connection
            connection))
        submit!
        (fn [request]
          (cluster/submit!
           router request
           (candidates/candidates (database-value) request [(observe)]
                                  {:chunk-size chunk-size})))
        cancel! (fn [request-id] (cluster/cancel-request! router request-id))
        http
        (try
          (openai-server/open-server
           (cond-> {:models models
                    :tokenize-chat tokenize-chat
                    :decode-token decode-token
                    :decode-tokens decode-tokens
                    :submit! submit!
                    :cancel! cancel!
                    :server-options (or server-options {})}
             max-pending-events (assoc :max-pending-events max-pending-events)))
          (catch Throwable error
            (try (.close ^Closeable router)
                 (finally (.close ^Closeable controller)))
            (throw error)))]
    (reset! http-holder http)
    (->LocalOpenAIServer http router controller nil connection nil false nil
                         nil nil (atom false))))

(def always-checkpoint-policy
  "A checkpoint policy that admits every completed continuation.

  Its cost fields are placeholders, not measurements. It is the default when a
  `:cache-directory` is configured so that a private server persists what it
  computes; replace it with values from
  `pretrained.continuation.benchmark/cache-policy-calibration` once measured."
  {:expected-reuses 1.0
   :checkpoint-ms 0.0
   :saved-ms-per-reuse 1.0})

(defn- resolve-model-directory
  [{:keys [model model-directory]}]
  (cond
    model-directory (str model-directory)
    model (let [entry (get lm/registry model)]
            (when-not entry
              (throw (ex-info "Unknown model registry key"
                              {:model model :known (lm/registered)})))
            (when-let [arch (:arch entry)] (require arch))
            ((requiring-resolve 'pretrained.hub/ensure-model) (:hf entry)))
    :else (throw (ex-info "Local server requires :model or :model-directory"
                          {}))))

(defn- catalog-config
  [cache-directory]
  (if cache-directory
    {:store {:backend :file :path (str (File. (str cache-directory) "catalog"))}
     :schema-flexibility :write :keep-history? false :value-caps :default}
    {:store {:backend :memory :id (random-uuid)}
     :schema-flexibility :write :keep-history? false :value-caps :default}))

(defn open-server
  "Load a model and serve it over `POST /v1/chat/completions` with durable KV.

  Options:

  - `:model`, a `pretrained.lm` registry key, or `:model-directory`, a local
    Hugging Face checkpoint directory. One is required.
  - `:model-id`, the public model id (default: the registry key name or the
    directory name).
  - `:port` (default 8080) and `:server-options` for HTTP-kit.
  - `:cache-directory`. When set, the Datahike catalog and chunk files live
    there and survive restarts, and completed continuations are checkpointed
    under `always-checkpoint-policy` unless `:checkpoint-policy` is given.
    When absent, the catalog is in memory and chunks go to a temporary
    directory removed on close.
  - `:device-id` (default `:ze:0`; use `:ocl:0` for OpenCL devices),
    `:max-position` (default 1024), `:page-size` (default 16),
    `:physical-pages` (default four sequences), and `:chunk-size` (default
    16). A resident route is advertised for reuse at its exact end and at
    every chunk boundary, so the default matches the page size; raise it to
    reduce durable chunk count for long contexts at the cost of coarser
    partial reuse.
  - `:execution-variant` (default `:gpu-q4k-paged`) and optional
    `:weights-id`, both part of the compatibility fingerprint.
  - `:tokenize-chat` to override the built-in chat template, and `:eos-ids`
    to override the checkpoint's stop tokens.
  - `:measurements` for the scheduler, defaulting to deployment constants.

  Returns a `Closeable` server. `local-port` reports the bound port."
  [{:keys [model-id port cache-directory device-id max-position page-size
           physical-pages chunk-size execution-variant weights-id tokenize-chat
           eos-ids checkpoint-policy measurements server-options
           offer-timeout-ms max-pending-events on-error!]
    :or {port 8080
         chunk-size 16
         execution-variant :gpu-q4k-paged}
    :as opts}]
  (let [directory (resolve-model-directory opts)
        model-id (or model-id
                     (some-> (:model opts) name)
                     (.getName (File. directory)))
        datahike-config (catalog-config cache-directory)
        ephemeral? (nil? cache-directory)
        temp-directory (when ephemeral?
                         (Files/createTempDirectory
                          "pretrained-local-" (make-array FileAttribute 0)))
        chunk-directory (if ephemeral?
                          (str temp-directory)
                          (str (File. (str cache-directory) "chunks")))
        connection (volatile! nil)
        worker (volatile! nil)]
    (try
      (vreset! connection (catalog/ensure-database! datahike-config))
      (let [model (loader/from-pretrained directory)
            fingerprint (model-identity/compatibility-fingerprint
                         model (cond-> {:execution-variant execution-variant}
                                 weights-id (assoc :weights-id weights-id)))
            quantized-weights (decoder-gpu/gpu-quantize model)
            ;; The paged pool stores FP16 rows.
            _ (model-identity/require-cache-dtype! execution-variant :float16)
            contract (model-identity/numerical-contract execution-variant)
            {:keys [tok encode decode]} (:tokenizer model)
            render (chat/render-fn directory)
            tokenize-chat (or tokenize-chat
                              (fn [messages] (vec (encode tok (render messages)))))]
        (vreset! worker
                 (model-worker/open-worker!
                  model quantized-weights
                  (cond-> {:fingerprint fingerprint
                           :connection @connection
                           :cache-directory chunk-directory
                           :chunk-size chunk-size
                           :checkpoint-policy
                           (or checkpoint-policy
                               (when-not ephemeral? always-checkpoint-policy))}
                    device-id (assoc :device-id device-id)
                    max-position (assoc :max-position max-position)
                    page-size (assoc :page-size page-size)
                    physical-pages (assoc :physical-pages physical-pages)
                    eos-ids (assoc :eos-ids eos-ids)
                    measurements (assoc :measurements measurements)
                    contract (assoc :numerical-contract contract))))
        (let [server
              (open-server-with-worker
               @connection
               (cond-> {:pool (:pool @worker)
                        :worker-opts (:worker-opts @worker)
                        :handlers (:handlers @worker)
                        :measurements (:measurements @worker)
                        :models {model-id fingerprint}
                        :tokenize-chat tokenize-chat
                        :decode-token #(decode tok [%])
                        :decode-tokens #(decode tok %)
                        :server-options (merge {:port port} server-options)
                        :chunk-size chunk-size}
                 offer-timeout-ms (assoc :offer-timeout-ms offer-timeout-ms)
                 max-pending-events (assoc :max-pending-events
                                           max-pending-events)
                 on-error! (assoc :on-error! on-error!)))]
          (assoc server
                 :worker @worker
                 :connection @connection
                 :datahike-config datahike-config
                 :ephemeral? ephemeral?
                 :temp-directory temp-directory
                 :model-id model-id
                 :fingerprint fingerprint)))
      (catch Throwable error
        (try (when-let [w @worker] (.close ^Closeable w)) (catch Throwable _))
        (try (when-let [c @connection] (d/release c)) (catch Throwable _))
        (when ephemeral?
          (try (d/delete-database datahike-config) (catch Throwable _))
          (when temp-directory (delete-tree! temp-directory)))
        (throw error)))))

(defn local-port
  "Return the OpenAI HTTP listener's bound port."
  [server]
  (openai-server/local-port (:http server)))

(defn router-state
  "Return the immutable cluster-router state."
  [server]
  (cluster/state (:router server)))

(defn worker-state
  "Return the immutable worker-machine state."
  [server]
  (local/state (:controller server)))
