(ns pretrained.distributed-continuation-demo
  "Executable two-worker continuation topology.

  The catalog has one authoritative Datahike writer. Workers connect through
  Kabel, keep independent catalog and tensor filestores, and exchange immutable
  tensor chunks through an S3-compatible Konserve store. Run with the
  `:distributed-demo` alias; MinIO is sufficient for local experiments."
  (:require [clojure.core.async :refer [<!!]]
            [datahike.api :as d]
            [datahike.kabel.cbor-handlers :as cbor-handlers]
            [datahike.kabel.connector]
            [datahike.kabel.handlers :as handlers]
            [is.simm.distributed-scope :as distributed-scope]
            [kabel.http-kit :refer [create-http-kit-handler!]]
            [kabel.peer :as peer]
            [konserve.store :as konserve-store]
            [konserve.tiered :as tiered]
            [konserve-s3.core]
            [konserve-sync.core :as sync]
            [pretrained.continuation :as continuation]
            [pretrained.continuation.catalog :as catalog]
            [pretrained.continuation.chunk-store :as chunk-store]
            [pretrained.continuation.gpu :as continuation-gpu]
            [pretrained.continuation.manager :as manager]
            [pretrained.continuation.paged-decoder :as paged-decoder]
            [pretrained.continuation.placement :as placement]
            [pretrained.continuation.replica :as replica]
            [superv.async :refer [<?? S]])
  (:import [java.io Closeable]
           [java.net ServerSocket]
           [java.nio.file Files]
           [java.util Comparator UUID]
           [java.util.concurrent CompletableFuture TimeUnit]))

(defn minio-config
  "Return a Konserve-S3 configuration for a local MinIO instance.

  `store-id` isolates the chunk namespace. The bucket defaults to
  `pretrained-continuations`; MinIO must be listening on `localhost:9000`."
  ([^UUID store-id]
   (minio-config store-id "pretrained-continuations"))
  ([^UUID store-id ^String bucket]
   {:backend :s3
    :region "us-east-1"
    :bucket bucket
    :id store-id
    :access-key "minioadmin"
    :secret "minioadmin"
    :path-style-access? true
    :endpoint-override {:protocol :http :hostname "localhost" :port 9000}}))

(defn connect-or-create-store!
  "Connect to `config`, creating the Konserve store when it is absent.

  The caller owns the returned store and should release it with
  `release-object-store!`. The bucket itself is not deleted."
  [config]
  (if (konserve-store/store-exists? config {:sync? true})
    (konserve-store/connect-store config {:sync? true})
    (konserve-store/create-store config {:sync? true})))

(defn release-object-store!
  "Release a store returned by `connect-or-create-store!` without deleting it."
  [config store]
  (konserve-store/release-store config store {:sync? true}))

(defn- free-port
  []
  (with-open [socket (ServerSocket. 0)]
    (.getLocalPort socket)))

(defn- delete-tree!
  [directory]
  (when (Files/exists directory (make-array java.nio.file.LinkOption 0))
    (with-open [paths (Files/walk directory
                                  (make-array java.nio.file.FileVisitOption 0))]
      (doseq [path (iterator-seq
                    (.iterator (.sorted paths (Comparator/reverseOrder))))]
        (Files/deleteIfExists path)))))

(defrecord Authority [url server-id store-id config connection server-peer]
  Closeable
  (close [_]
    (handlers/unregister-store-for-remote-access! store-id server-peer)
    (<?? S (peer/stop server-peer))
    (d/release connection)))

(defn open-authority!
  "Start the authoritative Datahike/Kabel catalog writer.

  `directory` is a durable Datahike filestore. `opts` may supply stable
  `:server-id`, `:store-id`, and `:port` values. The database and its files are
  retained on close so the authority can be restarted."
  ([directory]
   (open-authority! directory {}))
  ([directory {:keys [server-id store-id port]
               :or {server-id (random-uuid)
                    store-id (random-uuid)}}]
   (let [port (or port (free-port))
         url (str "ws://localhost:" port)
         config {:store {:backend :file :path (str directory) :id store-id}
                 :schema-flexibility :write
                 :keep-history? true
                 :value-caps :default}
         connection (catalog/ensure-database! config)
         handler (create-http-kit-handler! S url server-id)
         server-peer (peer/server-peer
                      S handler server-id
                      (comp (sync/server-middleware)
                            distributed-scope/remote-middleware)
                      cbor-handlers/datahike-cbor-middleware)]
     (<?? S (peer/start server-peer))
     (distributed-scope/invoke-on-peer server-peer)
     (handlers/register-global-handlers! server-peer)
     (handlers/register-store-for-remote-access!
      store-id connection server-peer {:branches :trunk})
     (->Authority url server-id store-id config connection server-peer))))

(defrecord Worker [node authority peer-id client-peer config connection cache
                   promoter executor]
  Closeable
  (close [_]
    (when executor
      (replica/close-executor! executor))
    (.close ^Closeable cache)
    (sync/unsubscribe-store! client-peer (:store-id authority))
    (<?? S (peer/stop client-peer))
    (d/release connection)))

(defn open-worker!
  "Connect one worker and start its tx-driven SSD replica reconciler.

  `catalog-directory` and `cache-directory` are worker-local durable paths.
  `chunk-backend-store` is a caller-owned shared Konserve store. A restarted
  worker may reuse both paths; a different cache path exercises a cold worker.
  Set `:reconcile? false` only when an external scheduler directly drives the
  returned worker's promoter, as the single-JVM smoke does."
  [^Authority authority ^String node catalog-directory cache-directory
   chunk-backend-store opts]
  (let [peer-id (or (:peer-id opts) (random-uuid))
        client-peer (peer/client-peer
                     S peer-id
                     (comp (sync/client-middleware)
                           distributed-scope/remote-middleware)
                     cbor-handlers/datahike-cbor-middleware)
        _ (distributed-scope/invoke-on-peer client-peer)
        _ (<?? S (peer/connect S client-peer (:url authority)))
        config {:store {:backend :file
                        :path (str catalog-directory)
                        :id (:store-id authority)}
                :index :datahike.index/persistent-set
                :schema-flexibility :write
                :keep-history? true
                :value-caps :default
                :writer {:backend :kabel
                         :peer-id (:server-id authority)
                         :store-id (:store-id authority)
                         :local-peer client-peer}}
        connection (<!! (d/connect config {:sync? false}))
        _ (when (instance? Throwable connection) (throw connection))
        cache (manager/open-manager
               nil cache-directory
               {:connection connection
                :chunk-size (or (:chunk-size opts) 256)
                :chunk-backend-store chunk-backend-store
                :max-pending-captures (or (:max-pending-captures opts) 2)
                :max-pending-publications (or (:max-pending-publications opts) 2)})
        promotion-store
        (tiered/connect-tiered-store
         (manager/local-chunk-store cache) chunk-backend-store
         :write-policy :frontend-only
         :read-policy :frontend-first
         :opts {:sync? true})
        promoter (replica/konserve-tiered-promoter promotion-store)
        executor (when (get opts :reconcile? true)
                   (replica/open-executor connection node :ssd promoter))]
    (->Worker node authority peer-id client-peer config connection cache
              promoter executor)))

(defn- await-until
  [timeout-ms operation]
  (let [deadline (+ (System/nanoTime) (* 1000000 (long timeout-ms)))]
    (loop []
      (if-let [value (operation)]
        value
        (if (< (System/nanoTime) deadline)
          (do (Thread/sleep 10) (recur))
          (throw (ex-info "Timed out waiting for distributed continuation state"
                          {:timeout-ms timeout-ms})))))))

(defn request-prefix!
  "Request all chunks ending at `published-chunks` for `worker`'s local SSD.

  Returns only after every requested replica is content-verified and announced
  ready through the worker's Kabel connection."
  ([^Worker worker model-fingerprint published-chunks]
   (request-prefix! worker model-fingerprint published-chunks 30000))
  ([^Worker worker model-fingerprint published-chunks timeout-ms]
   (let [prefixes (mapv :chunk/prefix-hash published-chunks)]
     (doseq [prefix prefixes]
       (await-until
        timeout-ms
        #(catalog/lookup-chunk @(:connection worker) model-fingerprint prefix))
       (placement/request!
        (:connection worker)
        {:model-fingerprint model-fingerprint
         :prefix-hash prefix
         :node (:node worker)
         :tier :ssd
         :priority 100}))
     (mapv
      (fn [prefix]
        (await-until
         timeout-ms
         #(some (fn [candidate]
                  (when (and (= (:node worker) (:kv/replica-node candidate))
                             (= :kv.replica/ready (:kv/replica-state candidate)))
                    candidate))
                (placement/replicas @(:connection worker)
                                    model-fingerprint prefix))))
      prefixes))))

(defn- promote-prefix-direct!
  [connection ^Worker worker model-fingerprint published-chunks]
  (mapv
   (fn [chunk]
     (let [prefix (:chunk/prefix-hash chunk)
           _ (placement/request!
              connection
              {:model-fingerprint model-fingerprint
               :prefix-hash prefix
               :node (:node worker)
               :tier :ssd
               :priority 100})
           action (some #(when (= prefix
                                  (get-in % [:demand :kv/demand-prefix-hash]))
                           %)
                        (:actions
                         (placement/reconciliation-plan
                          @connection (:node worker))))
           _ (when-not action
               (throw (ex-info "Placement policy did not produce a promotion action"
                               {:prefix-hash prefix})))
           result (replica/promote-action!
                   connection (:node worker) :ssd (:promoter worker) action)]
       (when-not (= :ready (:status result))
         (throw (:error result)))
       (some #(when (and (= (:node worker) (:kv/replica-node %))
                         (= :kv.replica/ready (:kv/replica-state %)))
                %)
             (placement/replicas @connection model-fingerprint prefix))))
   published-chunks))

(defn run-minio-smoke!
  "Run a self-cleaning, model-free two-worker handoff against local MinIO.

  The workers run sequentially because distributed-scope routes one connection
  per remote peer inside a JVM; production workers are separate processes. The
  smoke additionally closes and reopens the destination on the same local paths
  to verify restart behavior. It creates a unique S3 store and temporary
  directories, deleting both after the result is assembled."
  []
  (let [root (Files/createTempDirectory
              "pretrained-distributed-smoke-"
              (make-array java.nio.file.attribute.FileAttribute 0))
        catalog-dir (.resolve root "catalog-writer")
        source-catalog (.resolve root "source-catalog")
        source-cache (.resolve root "source-cache")
        destination-catalog (.resolve root "destination-catalog")
        destination-cache (.resolve root "destination-cache")
        store-config (minio-config (random-uuid))
        object-store (connect-or-create-store! store-config)
        authority (open-authority! catalog-dir)
        source (atom
                (open-worker! authority "worker-a" source-catalog source-cache
                              object-store {:chunk-size 2}))
        destination (atom nil)
        model {:n-layers 1 :n-kv 1 :head-dim 2}
        state {:continuation/backend :cpu
               :continuation/model model
               :continuation/model-fingerprint "distributed-smoke-v1"
               :continuation/layout
               (continuation/model-layout model)
               :continuation/max-position 8
               :continuation/processed-count 4
               :continuation/pending-token 5
               :continuation/tokens [1 2 3 4 5]
               :continuation/keys
               [(float-array [10 11 20 21 30 31 40 41 0 0 0 0 0 0 0 0])]
               :continuation/values
               [(float-array [50 51 60 61 70 71 80 81 0 0 0 0 0 0 0 0])]}]
    (try
      (let [started (System/nanoTime)
            published (manager/checkpoint-cpu-chunks! (:cache @source) state)
            publish-done (System/nanoTime)
            source-stats (manager/stats (:cache @source))
            _ (.close ^Closeable @source)
            _ (reset! source nil)
            first-destination
            (open-worker! authority "worker-b"
                          destination-catalog destination-cache
                          object-store {:chunk-size 2 :reconcile? false})
            _ (reset! destination first-destination)
            ready (promote-prefix-direct!
                   (:connection authority) first-destination
                   "distributed-smoke-v1" published)
            promotion-done (System/nanoTime)
            element-counts
            (mapv (fn [entry]
                    (chunk-store/with-mmap-payload
                     (manager/local-chunk-store (:cache first-destination))
                     (:kv/replica-store-key entry)
                     :element-count))
                  ready)
            _ (.close ^Closeable @destination)
            _ (reset! destination nil)
            restarted (open-worker! authority "worker-b"
                                    destination-catalog destination-cache
                                    object-store {:chunk-size 2 :reconcile? false})
            _ (reset! destination restarted)
            lookup (manager/lookup-chunk-prefix
                    (:cache restarted) "distributed-smoke-v1" [1 2 3 4 5])
            partial-lookup (manager/lookup-chunk-prefix
                            (:cache restarted) "distributed-smoke-v1"
                            [1 2 3 4 99 100])
            warm-started (System/nanoTime)
            warm-element-counts
            (mapv (fn [entry]
                    (chunk-store/with-mmap-payload
                     (manager/local-chunk-store (:cache restarted))
                     (:kv/store-key entry)
                     :element-count))
                  (:matched lookup))
            warm-done (System/nanoTime)]
        {:chunks (count published)
         :ready-replicas (count ready)
         :mmap-element-counts element-counts
         :timing-ms {:backend+catalog (/ (- publish-done started) 1.0e6)
                     :promotion (/ (- promotion-done publish-done) 1.0e6)}
         :source-cache-stats source-stats
         :restart {:cached-token-count (:cached-token-count lookup)
                   :matched-chunks (count (:matched lookup))
                   :warm-mmap-element-counts warm-element-counts
                   :warm-mmap-ms (/ (- warm-done warm-started) 1.0e6)}
         :partial-hit {:requested-token-count 5
                       :cached-token-count (:cached-token-count partial-lookup)
                       :matched-chunks (count (:matched partial-lookup))}})
      (finally
        (when-let [worker @destination]
          (.close ^Closeable worker))
        (when-let [worker @source]
          (.close ^Closeable worker))
        (.close ^Closeable authority)
        (release-object-store! store-config object-store)
        (konserve-store/delete-store store-config {:sync? true})
        (d/delete-database (:config authority))
        (delete-tree! root)))))

(defn checkpoint-gpu-prefix!
  "Checkpoint a GPU prompt and return a serializable destination manifest.

  `opts` requires `:model-fingerprint` and accepts `:tail-tokens` (default 1).
  Reference tokens are generated after publication for comparison with a
  destination process. Payload arrays and live GPU state are not returned."
  [^Worker source dstate prompt-ids
   {:keys [model-fingerprint tail-tokens] :or {tail-tokens 1}}]
  (when-not (string? model-fingerprint)
    (throw (ex-info "checkpoint-gpu-prefix! requires a model fingerprint" {})))
  (let [started (System/nanoTime)
        source-state (continuation-gpu/start-gpu
                      dstate prompt-ids
                      {:model-fingerprint model-fingerprint})
        ticket (manager/checkpoint-gpu-chunks-async! (:cache source) source-state)
        _ (when-not (:accepted? ticket)
            (throw (ex-info "Source continuation checkpoint was rejected" {})))
        submitted (System/nanoTime)
        captured (.get ^CompletableFuture (:captured ticket) 120 TimeUnit/SECONDS)
        capture-done (System/nanoTime)
        published (.get ^CompletableFuture (:published ticket) 120 TimeUnit/SECONDS)
        publish-done (System/nanoTime)
        reference (continuation-gpu/advance-gpu source-state tail-tokens)
        complete (System/nanoTime)
        milliseconds (fn [a b] (/ (- b a) 1.0e6))
        manifest-keys [:chunk/index :chunk/start :chunk/token-count
                       :chunk/parent-hash :chunk/prefix-hash :store-key :bytes]]
    {:model-fingerprint model-fingerprint
     :prompt-ids (vec prompt-ids)
     :tail-tokens (long tail-tokens)
     :reference-tokens (:tokens reference)
     :manifest (mapv #(select-keys % manifest-keys) published)
     :chunks (count captured)
     :timing-ms {:source-prefill+submit (milliseconds started submitted)
                 :capture (milliseconds submitted capture-done)
                 :backend+catalog (milliseconds capture-done publish-done)
                 :reference-decode (milliseconds publish-done complete)}
     :cache-stats (manager/stats (:cache source))}))

(defn restore-gpu-prefix!
  "Promote a source manifest, restore its prompt, and resume on this worker.

  `checkpoint` is the value returned by `checkpoint-gpu-prefix!` in another
  process. The result includes exactness against its reference token sequence
  and separate promotion, restore, and decode timings."
  [^Worker destination dstate checkpoint]
  (let [{:keys [model-fingerprint prompt-ids tail-tokens
                reference-tokens manifest]} checkpoint
        started (System/nanoTime)
        replicas (request-prefix! destination model-fingerprint manifest)
        promotion-done (System/nanoTime)
        restored (manager/restore-gpu-prefix
                  (:cache destination) dstate model-fingerprint prompt-ids)
        restore-done (System/nanoTime)
        resumed (continuation-gpu/advance-gpu
                 (:continuation restored) tail-tokens)
        complete (System/nanoTime)
        milliseconds (fn [a b] (/ (- b a) 1.0e6))]
    {:token-exact? (= reference-tokens (:tokens resumed))
     :reference-tokens reference-tokens
     :resumed-tokens (:tokens resumed)
     :cached-token-count (:cached-token-count restored)
     :replicas (count replicas)
     :timing-ms {:promotion (milliseconds started promotion-done)
                 :restore (milliseconds promotion-done restore-done)
                 :decode (milliseconds restore-done complete)}
     :cache-stats (manager/stats (:cache destination))}))

(defn checkpoint-paged-prefix!
  "Checkpoint a paged prompt and return a serializable destination manifest.

  `decoder` is a worker-local paged decoder. `opts` requires
  `:model-fingerprint` and accepts `:continuation-id` plus `:tail-tokens`
  (default 1). Prompt priming computes K/V through the penultimate token, the
  asynchronous capture publishes that exact prefix, and reference generation
  then continues on the same resident route."
  [^Worker source decoder prompt-ids
   {:keys [model-fingerprint continuation-id tail-tokens]
    :or {continuation-id :source-continuation tail-tokens 1}}]
  (when-not (string? model-fingerprint)
    (throw (ex-info "checkpoint-paged-prefix! requires a model fingerprint" {})))
  (let [prompt-ids (vec prompt-ids)
        started (System/nanoTime)
        _ (paged-decoder/prime-prompt! decoder continuation-id prompt-ids)
        ticket (manager/checkpoint-paged-chunks-async!
                (:cache source) (:pool decoder) continuation-id
                model-fingerprint prompt-ids)
        _ (when-not (:accepted? ticket)
            (throw (ex-info "Source paged checkpoint was rejected" {})))
        submitted (System/nanoTime)
        captured (.get ^CompletableFuture (:captured ticket) 120 TimeUnit/SECONDS)
        capture-done (System/nanoTime)
        published (.get ^CompletableFuture (:published ticket) 120 TimeUnit/SECONDS)
        publish-done (System/nanoTime)
        reference-tokens (paged-decoder/generate!
                          decoder continuation-id prompt-ids tail-tokens)
        complete (System/nanoTime)
        milliseconds (fn [a b] (/ (- b a) 1.0e6))
        manifest-keys [:chunk/index :chunk/start :chunk/token-count
                       :chunk/parent-hash :chunk/prefix-hash :store-key :bytes]]
    {:model-fingerprint model-fingerprint
     :prompt-ids prompt-ids
     :tail-tokens (long tail-tokens)
     :reference-tokens reference-tokens
     :manifest (mapv #(select-keys % manifest-keys) published)
     :chunks (count captured)
     :timing-ms {:source-prefill+submit (milliseconds started submitted)
                 :capture (milliseconds submitted capture-done)
                 :backend+catalog (milliseconds capture-done publish-done)
                 :reference-decode (milliseconds publish-done complete)}
     :cache-stats (manager/stats (:cache source))}))

(defn restore-paged-prefix!
  "Promote, restore, and resume a paged checkpoint on another worker.

  The destination decoder owns a distinct worker-local GPU page pool. `opts`
  accepts its `:continuation-id`; tensor chunks move through the shared object
  store and local mmap frontend, while only the catalog and placement facts go
  through Kabel/Datahike."
  ([^Worker destination decoder checkpoint]
   (restore-paged-prefix! destination decoder checkpoint
                          {:continuation-id :destination-continuation}))
  ([^Worker destination decoder checkpoint {:keys [continuation-id]
                                             :or {continuation-id
                                                  :destination-continuation}}]
   (let [{:keys [model-fingerprint prompt-ids tail-tokens
                 reference-tokens manifest]} checkpoint
         started (System/nanoTime)
         replicas (request-prefix! destination model-fingerprint manifest)
         promotion-done (System/nanoTime)
         restored (manager/restore-paged-prefix!
                   (:cache destination) (:pool decoder) continuation-id
                   model-fingerprint prompt-ids)
         restore-done (System/nanoTime)
         resumed-tokens (paged-decoder/generate!
                         decoder continuation-id prompt-ids tail-tokens)
         complete (System/nanoTime)
         milliseconds (fn [a b] (/ (- b a) 1.0e6))]
     {:token-exact? (= reference-tokens resumed-tokens)
      :reference-tokens reference-tokens
      :resumed-tokens resumed-tokens
      :cached-token-count (:cached-token-count restored)
      :replicas (count replicas)
      :timing-ms {:promotion (milliseconds started promotion-done)
                  :restore (milliseconds promotion-done restore-done)
                  :decode (milliseconds restore-done complete)}
      :cache-stats (manager/stats (:cache destination))})))
