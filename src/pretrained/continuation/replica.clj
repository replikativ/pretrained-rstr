(ns pretrained.continuation.replica
  "Off-band transfer and bounded reconciliation of immutable KV chunks.

  Datahike remains the control plane. Promoters make one immutable chunk local
  outside transactions, verify it, and only then announce a ready replica. The
  built-in promoter explicitly warms a Konserve tiered store's filestore
  frontend; mmap-based GPU restoration continues to use that concrete frontend."
  (:require [konserve.core :as k]
            [konserve.tiered :as tiered]
            [pretrained.continuation.chunk-store :as chunk-store]
            [pretrained.continuation.placement :as placement])
  (:import [java.io Closeable]
           [java.util.concurrent ArrayBlockingQueue ExecutorService
            RejectedExecutionException ThreadFactory ThreadPoolExecutor
            ThreadPoolExecutor$AbortPolicy TimeUnit]
           [java.util.concurrent.atomic AtomicBoolean AtomicLong]))

(declare verify-chunk!)

(defprotocol ReplicaPromoter
  "The narrow data-plane effect behind declarative replica placement."
  (ensure-local! [promoter action]
    "Make `action`'s chunk durably local and verified.

    Returns `{:store-key :path :bytes}` only after the target can serve the
    immutable content. Implementations must throw on missing or corrupt data."))

(defrecord KonserveTieredPromoter [frontend-store backend-store]
  ReplicaPromoter
  (ensure-local! [_ action]
    (let [catalog-chunk (:chunk action)
          store-key (or (:kv/blob catalog-chunk)
                        (:kv/store-key catalog-chunk))]
      (when-not (chunk-store/stored? frontend-store store-key)
        (tiered/sync-keys-to-frontend!
         frontend-store backend-store [store-key] {:sync? true}))
      (try
        (let [decoded (chunk-store/read-chunk frontend-store store-key)]
          (verify-chunk! catalog-chunk store-key decoded)
          (chunk-store/describe frontend-store store-key))
        (catch Throwable error
          (try
            (k/dissoc frontend-store store-key {:sync? true})
            (catch Throwable invalidation-error
              (.addSuppressed error invalidation-error)))
          (throw error))))))

(defn konserve-tiered-promoter
  "Create a promoter that explicitly warms `tiered-store`'s local frontend.

  The operation waits for targeted backend-to-frontend synchronization before
  returning. The frontend must be an uncompressed Boring filestore so the
  resulting chunk can be verified and subsequently memory-mapped."
  [tiered-store]
  (let [frontend-store (:frontend-store tiered-store)
        backend-store (:backend-store tiered-store)]
    (when-not (and frontend-store backend-store)
      (throw (ex-info "A Konserve tiered promoter requires both store tiers"
                      {:store tiered-store})))
    (->KonserveTieredPromoter frontend-store backend-store)))

(defn- verify-chunk!
  [catalog-chunk expected-store-key decoded]
  (when-not decoded
    (throw (ex-info "Replica source does not contain its announced chunk"
                    {:store-key expected-store-key})))
  (let [actual-store-key (chunk-store/content-id decoded)
        expected-fields {:chunk/model-fingerprint (:kv/model-fingerprint catalog-chunk)
                         :chunk/prefix-hash (:kv/prefix-hash catalog-chunk)
                         :chunk/start (:kv/start-token catalog-chunk)
                         :chunk/token-count (:kv/token-count catalog-chunk)}
        actual-fields (select-keys decoded (keys expected-fields))]
    (when-not (= expected-store-key actual-store-key)
      (throw (ex-info "Replica chunk failed content verification"
                      {:expected expected-store-key :actual actual-store-key})))
    (when-not (= expected-fields actual-fields)
      (throw (ex-info "Replica chunk does not match its catalog entry"
                      {:expected expected-fields :actual actual-fields})))
    decoded))

(defn- replica-announcement
  [node tier state action & storage]
  (let [chunk (:chunk action)]
    (merge {:model-fingerprint (:kv/model-fingerprint chunk)
            :prefix-hash (:kv/prefix-hash chunk)
            :node node
            :tier tier
            :state state}
           (first storage))))

(defn promote-action!
  "Synchronously promote one planner action outside the inference path.

  The promoter owns the transport/storage operation and verification. This
  function owns Datahike lifecycle transitions and additionally checks that the
  promoter returned the catalog's content identity. Failures are announced and
  returned as `:error`."
  [connection node tier promoter action]
  (let [catalog-chunk (:chunk action)
        expected-store-key (or (:kv/blob catalog-chunk)
                               (:kv/store-key catalog-chunk))]
    (placement/announce-replica!
     connection (replica-announcement node tier :kv.replica/copying action))
    (try
      (let [stored (ensure-local! promoter action)]
        (when-not (= expected-store-key (:store-key stored))
          (throw (ex-info "Promoter returned a different chunk identity"
                          {:expected expected-store-key
                           :actual (:store-key stored)})))
        (placement/announce-replica!
         connection
         (replica-announcement node tier :kv.replica/ready action stored))
        {:status :ready :stored stored :action action})
      (catch Throwable error
        (placement/announce-replica!
         connection
         (replica-announcement node tier :kv.replica/failed action
                               {:store-key expected-store-key
                                :error (or (.getMessage error)
                                           (str (class error)))}))
        {:status :failed :error error :action action}))))

(declare close-executor!)

(defrecord ReplicaExecutor [connection node tier promoter worker
                            closed? listener-key failed-attempts metrics]
  Closeable
  (close [executor]
    (close-executor! executor)))

(defn- daemon-thread-factory
  []
  (let [counter (AtomicLong.)]
    (reify ThreadFactory
      (newThread [_ runnable]
        (doto (Thread. runnable
                       (str "pretrained-kv-replica-" (.incrementAndGet counter)))
          (.setDaemon true)
          (.setPriority Thread/MIN_PRIORITY))))))

(defn- attempt-key
  [action]
  [(:kv/demand-id (:demand action))
   (mapv :kv/replica-id (:sources action))])

(defn- reconcile*
  [^ReplicaExecutor executor]
  (let [actions (->> (:actions
                      (placement/reconciliation-plan
                       @(:connection executor) (:node executor)))
                     (filter #(= (:tier executor)
                                 (get-in % [:demand :kv/demand-tier])))
                     vec)
        results
        (mapv (fn [action]
                (let [key (attempt-key action)]
                  (if (contains? @(:failed-attempts executor) key)
                    {:status :suppressed :reason :previous-failure :action action}
                    (let [result (promote-action!
                                  (:connection executor) (:node executor)
                                  (:tier executor) (:promoter executor) action)]
                      (when (= :failed (:status result))
                        (swap! (:failed-attempts executor) conj key))
                      result))))
              actions)]
    (swap! (:metrics executor)
           (fn [metrics]
             (reduce (fn [current result]
                       (update current (:status result) (fnil inc 0)))
                     (update metrics :reconciliations inc)
                     results)))
    results))

(defn reconcile-once!
  "Reconcile the executor's currently visible demands once, synchronously.

  This entry point is useful for deterministic tests and REPL operation. The
  executor's listener runs the same logic on its bounded background worker."
  [^ReplicaExecutor executor]
  (when (.get ^AtomicBoolean (:closed? executor))
    (throw (IllegalStateException. "Replica executor is closed")))
  (reconcile* executor))

(defn trigger!
  "Offer one reconciliation pass to the bounded worker without blocking.

  Returns true when accepted. A false result is safe: the running or queued pass
  reads the latest Datahike value and therefore coalesces this notification."
  [^ReplicaExecutor executor]
  (if (.get ^AtomicBoolean (:closed? executor))
    false
    (try
      (.execute ^ThreadPoolExecutor (:worker executor)
                ^Runnable (fn [] (reconcile* executor)))
      (swap! (:metrics executor) update :signals-accepted inc)
      true
      (catch RejectedExecutionException _
        (swap! (:metrics executor) update :signals-coalesced inc)
        false))))

(defn retry-failed!
  "Forget process-local failed-attempt suppression and trigger reconciliation."
  [^ReplicaExecutor executor]
  (reset! (:failed-attempts executor) #{})
  (trigger! executor))

(defn stats
  "Return process-local reconciliation counters and bounded queue depth."
  [^ReplicaExecutor executor]
  (assoc @(:metrics executor)
         :queue-depth (.size (.getQueue ^ThreadPoolExecutor (:worker executor)))
         :failed-attempts (count @(:failed-attempts executor))))

(defn close-executor!
  "Stop listening, drain accepted replication work, and stop the worker."
  [^ReplicaExecutor executor]
  (when (.compareAndSet ^AtomicBoolean (:closed? executor) false true)
    (placement/unlisten! (:connection executor) (:listener-key executor))
    (.shutdown ^ExecutorService (:worker executor))
    (when-not (.awaitTermination ^ExecutorService (:worker executor)
                                60 TimeUnit/SECONDS)
      (.shutdownNow ^ExecutorService (:worker executor))
      (.awaitTermination ^ExecutorService (:worker executor)
                         10 TimeUnit/SECONDS))))

(defn open-executor
  "Start a non-blocking tx-driven replica reconciler for one node and tier.

  `promoter` performs one verified storage/transport effect. `:max-pending`
  defaults to one. This intentionally coalesces transaction bursts instead of
  allowing cache traffic to create unbounded work or inference-path
  backpressure."
  ([connection node tier promoter]
   (open-executor connection node tier promoter {}))
  ([connection node tier promoter {:keys [max-pending]
                                    :or {max-pending 1}}]
   (when-not (pos? max-pending)
     (throw (ex-info "Replica queue capacity must be positive"
                     {:max-pending max-pending})))
   (let [listener-key (random-uuid)
         worker (ThreadPoolExecutor.
                 1 1 0 TimeUnit/MILLISECONDS
                 (ArrayBlockingQueue. (int max-pending))
                 (daemon-thread-factory)
                 (ThreadPoolExecutor$AbortPolicy.))
         executor (->ReplicaExecutor
                   connection node tier promoter worker
                   (AtomicBoolean. false) listener-key (atom #{})
                   (atom {:reconciliations 0 :signals-accepted 0
                          :signals-coalesced 0}))]
     (placement/listen! connection listener-key node
                        (fn [_transaction] (trigger! executor)))
     (trigger! executor)
     executor)))
