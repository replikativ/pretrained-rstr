(ns pretrained.continuation.replica
  "Off-band transfer and bounded reconciliation of immutable KV chunks.

  Datahike remains the control plane. Repositories move tensor bytes outside
  transactions, verify their content identity, and only then announce a ready
  replica. The built-in Konserve repository decodes during cross-store copies;
  mmap-based GPU restoration remains unchanged and copy-free on the CPU side."
  (:require [pretrained.continuation.chunk-store :as chunk-store]
            [pretrained.continuation.placement :as placement])
  (:import [java.io Closeable]
           [java.util.concurrent ArrayBlockingQueue ExecutorService
            RejectedExecutionException ThreadFactory ThreadPoolExecutor
            ThreadPoolExecutor$AbortPolicy TimeUnit]
           [java.util.concurrent.atomic AtomicBoolean AtomicLong]))

(defprotocol ChunkRepository
  "A worker-visible repository of immutable continuation chunks."
  (chunk-present? [repository store-key]
    "Return true when `store-key` is locally readable.")
  (read-replica-chunk [repository store-key]
    "Return the decoded immutable chunk, or nil when absent.")
  (write-replica-chunk! [repository chunk]
    "Durably store `chunk` and return `{:store-key :path :bytes}`."))

(defrecord KonserveRepository [store]
  ChunkRepository
  (chunk-present? [_ store-key]
    (chunk-store/stored? store store-key))
  (read-replica-chunk [_ store-key]
    (chunk-store/read-chunk store store-key))
  (write-replica-chunk! [_ chunk]
    (chunk-store/put! store chunk)))

(defn konserve-repository
  "Wrap a continuation chunk store as a replication repository."
  [store]
  (->KonserveRepository store))

(defn repository-resolver
  "Return a source resolver backed by a node or `[node tier]` repository map.

  This models local mounts and shared filesystems in one process. HTTP, S3, or
  peer-to-peer transports can instead provide a function with the same shape."
  [repositories]
  (fn [replica]
    (or (get repositories [(:kv/replica-node replica)
                           (:kv/replica-tier replica)])
        (get repositories (:kv/replica-node replica)))))

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

(defn copy-action!
  "Synchronously execute one planner action outside the inference path.

  `resolve-source` maps the action's source replica to a `ChunkRepository`.
  Bytes become visible at `target` before the ready announcement. Content or
  catalog mismatches announce a failed replica and are returned as `:error`.
  A missing source or unavailable adapter returns `:waiting` without claiming a
  copy attempt."
  [connection node tier target resolve-source action]
  (let [candidates (or (seq (:sources action))
                       (some-> (:source action) vector))
        resolution-error (atom nil)
        [source source-repository]
        (some (fn [candidate]
                (try
                  (when-let [repository (resolve-source candidate)]
                    [candidate repository])
                  (catch Throwable error
                    (reset! resolution-error error)
                    nil)))
              candidates)
        catalog-chunk (:chunk action)
        expected-store-key (or (:kv/blob catalog-chunk)
                               (:kv/store-key catalog-chunk))]
    (cond
      (empty? candidates)
      {:status :waiting :reason :no-source :action action}

      (nil? source-repository)
      (cond-> {:status :waiting :reason :source-unavailable :action action}
        @resolution-error (assoc :error @resolution-error))

      :else
      (do
        (placement/announce-replica!
         connection (replica-announcement node tier :kv.replica/copying action))
        (try
          (when-not (= expected-store-key
                       (or (:kv/replica-blob source)
                           (:kv/replica-store-key source)))
            (throw (ex-info "Source announcement names a different store key"
                            {:expected expected-store-key
                             :source (or (:kv/replica-blob source)
                                         (:kv/replica-store-key source))})))
          (let [read-repository (if (chunk-present? target expected-store-key)
                                  target
                                  source-repository)
                decoded (verify-chunk!
                         catalog-chunk expected-store-key
                         (read-replica-chunk read-repository expected-store-key))
                stored (write-replica-chunk! target decoded)]
            (when-not (= expected-store-key (:store-key stored))
              (throw (ex-info "Target repository changed the chunk identity"
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
            {:status :failed :error error :action action}))))))

(declare close-executor!)

(defrecord ReplicaExecutor [connection node tier target resolve-source worker
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
                    (let [result (copy-action!
                                  (:connection executor) (:node executor)
                                  (:tier executor) (:target executor)
                                  (:resolve-source executor) action)]
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

  `target` is the worker-local `ChunkRepository`; `resolve-source` maps ready
  replica facts to readable repositories. `:max-pending` defaults to one. This
  intentionally coalesces transaction bursts instead of allowing cache traffic
  to create unbounded work or inference-path backpressure."
  ([connection node tier target resolve-source]
   (open-executor connection node tier target resolve-source {}))
  ([connection node tier target resolve-source {:keys [max-pending]
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
                   connection node tier target resolve-source worker
                   (AtomicBoolean. false) listener-key (atom #{})
                   (atom {:reconciliations 0 :signals-accepted 0
                          :signals-coalesced 0}))]
     (placement/listen! connection listener-key node
                        (fn [_transaction] (trigger! executor)))
     (trigger! executor)
     executor)))
