(ns pretrained.continuation.manager
  "Local durable continuation manager: snapshot files plus a Datahike catalog."
  (:require [datahike.api :as d]
            [konserve.tiered :as tiered]
            [pretrained.continuation :as continuation]
            [pretrained.attention-state :as attention-state]
            [pretrained.continuation.catalog :as catalog]
            [pretrained.continuation.manifest :as manifest]
            [pretrained.continuation.parts :as parts]
            [pretrained.continuation.chunk :as chunk]
            [pretrained.continuation.chunk-store :as chunk-store]
            [pretrained.continuation.content-provider :as content-provider]
            [pretrained.continuation.gpu :as continuation-gpu]
            [pretrained.continuation.page-pool :as page-pool]
            [pretrained.continuation.residency :as residency]
            [pretrained.continuation.store :as store]
            [pretrained.continuation.telemetry :as telemetry]
            [raster.runtime.numerical-content :as content])
  (:import [java.io Closeable]
           [java.lang AutoCloseable]
           [java.nio.file Files StandardCopyOption]
           [java.util UUID]
           [java.util.concurrent ArrayBlockingQueue CancellationException CompletableFuture
            ExecutorService RejectedExecutionException ThreadFactory ThreadPoolExecutor
            ThreadPoolExecutor$AbortPolicy TimeUnit]
           [java.util.concurrent.atomic AtomicBoolean AtomicLong]))

(declare close-manager!)

(defrecord Manager [connection owns-connection? directory chunk-store chunk-write-store
                    content-provider chunk-size max-chunk-staging-bytes
                    calibration-alpha capture-executor publish-executor closed? metrics
                    numerical-contract layout-records]
  Closeable
  (close [manager] (close-manager! manager)))

(defn- daemon-thread-factory
  [prefix]
  (let [counter (AtomicLong.)]
    (reify ThreadFactory
      (newThread [_ runnable]
        (doto (Thread. runnable (str prefix (.incrementAndGet counter)))
          (.setDaemon true)
          (.setPriority Thread/MIN_PRIORITY))))))

(defn- bounded-executor
  [prefix capacity]
  (ThreadPoolExecutor. 1 1 0 TimeUnit/MILLISECONDS
                       (ArrayBlockingQueue. (int capacity))
                       (daemon-thread-factory prefix)
                       (ThreadPoolExecutor$AbortPolicy.)))

(defn- record-chunk-plan!
  [^Manager manager planned missing]
  (swap! (:metrics manager)
         (fn [metrics]
           (-> metrics
               (update :chunks-planned + (count planned))
               (update :chunks-reused + (- (count planned) (count missing)))
               (update :chunks-stored + (count missing))))))

(defn stats
  "Return cache outcomes and current bounded-worker queue depths.

  Counters are process-local and monotonic for this manager instance. They make
  cache value and inference-path backpressure visible without querying Datahike."
  [^Manager manager]
  (merge @(:metrics manager)
         (content-provider/stats (:content-provider manager))
         {:max-chunk-staging-bytes (:max-chunk-staging-bytes manager)
          :capture-queue-depth (.size (.getQueue ^ThreadPoolExecutor
                                                 (:capture-executor manager)))
          :publish-queue-depth (.size (.getQueue ^ThreadPoolExecutor
                                                 (:publish-executor manager)))}))

(defn- stop-executor!
  [^ExecutorService executor]
  (.shutdown executor)
  (when-not (.awaitTermination executor 60 TimeUnit/SECONDS)
    (.shutdownNow executor)
    (.awaitTermination executor 10 TimeUnit/SECONDS)))

(defn close-manager!
  "Drain accepted checkpoints, stop background workers and release Datahike.

  Capture drains before publication is stopped, ensuring an accepted capture can
  enqueue its catalog transaction during shutdown. Repeated calls are harmless."
  [^Manager manager]
  (when (.compareAndSet ^AtomicBoolean (:closed? manager) false true)
    (stop-executor! (:capture-executor manager))
    (stop-executor! (:publish-executor manager))
    (.close ^Closeable (:content-provider manager))
    (when (:owns-connection? manager)
      (d/release (:connection manager)))))

(defn connection
  "Return the Datahike connection used by `manager`.

  The manager owns and releases connections it opens from a configuration. A
  connection supplied to `open-manager` remains owned by the caller."
  [^Manager manager]
  (:connection manager))

(defn local-chunk-store
  "Return the manager's mmap-compatible local Konserve chunk store.

  This is the concrete frontend to use for worker-local promotion and direct
  mmap restoration. The manager retains ownership of the store lifecycle."
  [^Manager manager]
  (:chunk-store manager))

(defn numerical-content-provider
  "Return the Raster provider used for chunk localization and scoped mmap."
  [^Manager manager]
  (:content-provider manager))

(defn open-manager
  "Open a continuation manager rooted at `directory` and `datahike-config`.

  `opts` controls bounded background work with `:max-pending-captures`,
  `:max-pending-publications`, and `:max-concurrent-localizations` (all default
  to 2), `:chunk-size` (default 256 processed tokens), and
  `:max-chunk-staging-bytes` (default 256 MiB), and `:warm-catalog-query?`
  (default true). `:calibration-alpha` controls live EWMA smoothing and defaults
  to 0.2. Warming executes one empty-result chunk lookup while opening
  the manager so Datahike query compilation cannot extend the first accepted
  checkpoint's route lifetime. Paged checkpoints whose single
  host payload would exceed that byte limit are rejected before queueing.
  `:chunk-backend-store` optionally supplies a
  caller-owned authoritative Konserve store. Chunk writes then return after the
  local filestore frontend and publish to Datahike only after their write-behind
  receipts succeed. `:connection` optionally supplies an already connected,
  caller-owned Datahike connection, including a Kabel client connection; when
  present, `datahike-config` is ignored. `:numerical-contract` is the
  `{:mode :determinism}` of a structured execution variant
  (`model-identity/numerical-contract`); it is recorded with the fingerprint's
  durable layout and otherwise defaults to the dtype's mode and `:toleranced`. Each stage has one low-priority daemon
  worker. Queue saturation rejects cache work instead of blocking inference."
  ([datahike-config directory] (open-manager datahike-config directory {}))
  ([datahike-config directory {:keys [max-pending-captures max-pending-publications
                                      max-concurrent-localizations chunk-size
                                      max-chunk-staging-bytes chunk-backend-store connection
                                      warm-catalog-query? calibration-alpha
                                      numerical-contract]
                               :or {max-pending-captures 2
                                    max-pending-publications 2
                                    max-concurrent-localizations 2
                                    chunk-size chunk/default-chunk-size
                                    max-chunk-staging-bytes (* 256 1024 1024)
                                    warm-catalog-query? true
                                    calibration-alpha telemetry/default-alpha}}]
   (when-not (and (pos? max-pending-captures) (pos? max-pending-publications)
                  (pos? max-concurrent-localizations) (pos? chunk-size)
                  (integer? max-chunk-staging-bytes)
                  (pos? max-chunk-staging-bytes)
                  (number? calibration-alpha)
                  (< 0.0 (double calibration-alpha))
                  (<= (double calibration-alpha) 1.0))
     (throw (ex-info "Checkpoint capacities must be positive"
                     {:max-pending-captures max-pending-captures
                      :max-pending-publications max-pending-publications
                      :max-concurrent-localizations max-concurrent-localizations
                      :chunk-size chunk-size
                      :max-chunk-staging-bytes max-chunk-staging-bytes
                      :calibration-alpha calibration-alpha})))
   (let [path (.toPath (java.io.File. (str directory)))
         chunk-path (.resolve path "chunks")]
     (Files/createDirectories path (make-array java.nio.file.attribute.FileAttribute 0))
     (Files/createDirectories chunk-path (make-array java.nio.file.attribute.FileAttribute 0))
     (let [local-store (chunk-store/open-store chunk-path)
           write-store (if chunk-backend-store
                         (tiered/connect-tiered-store
                          local-store chunk-backend-store
                          :write-policy :write-behind
                          :read-policy :frontend-first
                          :opts {:sync? true})
                         local-store)
           catalog-connection (or connection
                                  (catalog/ensure-database! datahike-config))
           manager
           (->Manager catalog-connection
                      (nil? connection)
                      path
                      local-store
                      write-store
                      (content-provider/open-provider
                       local-store write-store
                       {:max-concurrent-localizations max-concurrent-localizations})
                      (long chunk-size)
                      (long max-chunk-staging-bytes)
                      (double calibration-alpha)
                      (bounded-executor "pretrained-kv-capture-" max-pending-captures)
                      (bounded-executor "pretrained-kv-publish-" max-pending-publications)
                      (AtomicBoolean. false)
                      (atom {:capture-accepted 0 :capture-rejected 0
                             :capture-byte-rejected 0
                             :chunks-planned 0 :chunks-reused 0 :chunks-stored 0
                             :prefix-lookups 0 :full-hits 0 :partial-hits 0 :misses 0
                             :requested-tokens 0 :cached-tokens 0
                             :restored-chunks 0 :restored-bytes 0})
                      numerical-contract
                      (atom {}))]
       (try
         (when warm-catalog-query?
           (catalog/lookup-chunks
            @catalog-connection "pretrained/catalog-query-warmup"
            [(UUID/fromString "00000000-0000-0000-0000-000000000000")]))
         manager
         (catch Throwable error
           (.close ^Closeable manager)
           (throw error)))))))

(defn- missing-chunk-descriptors
  [^Manager manager model-fingerprint descriptors]
  (let [present (catalog/lookup-chunks
                 @(:connection manager) model-fingerprint
                 (mapv :chunk/prefix-hash descriptors))
        present-hashes (into #{} (map :kv/prefix-hash) present)]
    (vec (remove #(contains? present-hashes (:chunk/prefix-hash %)) descriptors))))

(defn- store-part!
  [^Manager manager tensor-chunk]
  (if (identical? (:chunk-store manager) (:chunk-write-store manager))
    (chunk-store/put! (:chunk-store manager) tensor-chunk)
    (chunk-store/put-write-behind!
     (:chunk-store manager) (:chunk-write-store manager) tensor-chunk)))

(defn- persist-node!
  "Store the parts of one chain node and return its catalog-ready description.

  A single part without `:chunk/group` is today's node-level blob. Several
  parts become `:parts`, each with its group and storage fields."
  [^Manager manager parts]
  (let [parts (if (map? parts) [parts] (vec parts))
        head (dissoc (first parts) :chunk/payload :chunk/group :chunk/slabs
                     :chunk/elements-per-slab)]
    (if (and (= 1 (count parts)) (nil? (:chunk/group (first parts))))
      (merge (dissoc (first parts) :chunk/payload) (store-part! manager (first parts)))
      (assoc head :parts
             (mapv (fn [part]
                     (assoc (store-part! manager part) :group (:chunk/group part)))
                   parts)))))

(defn- persist-chunks!
  [^Manager manager nodes]
  (mapv #(persist-node! manager %) nodes))

(defn- node-store-keys
  [stored]
  (if-let [parts (:parts stored)]
    (mapv :store-key parts)
    [(:store-key stored)]))

(defn- node-bytes
  [stored]
  (if-let [parts (:parts stored)]
    (reduce + 0 (map :bytes parts))
    (:bytes stored)))

(defn- durable-node!
  [stored]
  (if (:parts stored)
    (update stored :parts #(mapv chunk-store/await-durable! %))
    (chunk-store/await-durable! stored)))

(defn- local-node
  [stored]
  (if (:parts stored)
    (update stored :parts #(mapv chunk-store/local-result %))
    (chunk-store/local-result stored)))

(defn- publish-chunks!
  [^Manager manager model-fingerprint stored-chunks]
  (let [durable-chunks (mapv durable-node! stored-chunks)]
    (catalog/put-chunks! (:connection manager) model-fingerprint durable-chunks
                         {:content-algorithm chunk-store/content-algorithm
                          :numerical-contract (:numerical-contract manager)})
    durable-chunks))

(defn checkpoint-cpu-chunks!
  "Persist missing immutable chunks from a CPU continuation and catalog them.

  Existing hash-chain nodes are reused. New Konserve objects are durable before
  their Datahike datoms are transacted together. Returns the newly stored chunks."
  [^Manager manager state]
  (let [plan (:chunks (chunk/continuation-plan state (:chunk-size manager)))
        missing (missing-chunk-descriptors
                 manager (:continuation/model-fingerprint state) plan)
        _ (record-chunk-plan! manager plan missing)
        stored (persist-chunks!
                manager (mapv #(chunk/cpu-tensor-chunks state %) missing))]
    (publish-chunks! manager (:continuation/model-fingerprint state) stored)))

(defn- submit-chunk-checkpoint!
  ([^Manager manager context]
   (submit-chunk-checkpoint! manager context {}))
  ([^Manager manager context {:keys [estimated-staging-bytes]}]
   (when-not (or (nil? estimated-staging-bytes)
                 (and (integer? estimated-staging-bytes)
                      (not (neg? estimated-staging-bytes))))
     (throw (ex-info "Estimated checkpoint staging bytes must be non-negative"
                     {:estimated-staging-bytes estimated-staging-bytes})))
   (let [captured (CompletableFuture.)
         published (CompletableFuture.)
         phase-timings (atom {})
         ticket (cond-> {:accepted? true :captured captured :published published
                         :phase-timings phase-timings}
                  estimated-staging-bytes
                  (assoc :estimated-staging-bytes (long estimated-staging-bytes)))
        reject! (fn [error]
                  (.completeExceptionally captured error)
                  (.completeExceptionally published error))
        capture-task
        (fn []
          (try
            (let [capture-started (System/nanoTime)
                  context-started (System/nanoTime)
                  {:keys [model-fingerprint plan export]} (context)
                  context-finished (System/nanoTime)
                  lookup-started context-finished
                  missing (missing-chunk-descriptors manager model-fingerprint plan)
                  lookup-finished (System/nanoTime)
                  _ (record-chunk-plan! manager plan missing)
                  export-ns (volatile! 0)
                  persistence-ns (volatile! 0)
                  mmap-preparation-ns (volatile! 0)
                  stored
                  (mapv (fn [descriptor]
                          (let [started (System/nanoTime)
                                tensor-parts (export descriptor)
                                exported (System/nanoTime)
                                stored (persist-node! manager tensor-parts)
                                persisted (System/nanoTime)
                                _ (doseq [store-key (node-store-keys stored)]
                                    (chunk-store/prepare-mmap-payload!
                                     (:chunk-store manager) store-key))
                                prepared (System/nanoTime)]
                            (vswap! export-ns + (- exported started))
                            (vswap! persistence-ns + (- persisted exported))
                            (vswap! mmap-preparation-ns + (- prepared persisted))
                            stored))
                        missing)
                  capture-finished (System/nanoTime)
                  capture-ms (/ (- capture-finished capture-started) 1.0e6)
                  stored-bytes (reduce + 0 (map node-bytes stored))
                  _ (reset! phase-timings
                            {:context-ms (/ (- context-finished context-started)
                                            1.0e6)
                             :catalog-lookup-ms
                             (/ (- lookup-finished lookup-started) 1.0e6)
                             :device-export-ms (/ @export-ns 1.0e6)
                             :local-persistence-ms (/ @persistence-ns 1.0e6)
                             :mmap-preparation-ms (/ @mmap-preparation-ns 1.0e6)
                             :capture-total-ms capture-ms
                             :planned-chunks (count plan)
                             :stored-chunks (count missing)})
                  _ (swap! (:metrics manager)
                           (fn [metrics]
                             (cond->
                              (update metrics :calibration telemetry/record
                                      :checkpoint-ms capture-ms
                                      (:calibration-alpha manager))
                               (pos? stored-bytes)
                               (update :calibration telemetry/record
                                       :checkpoint-ms-per-byte
                                       (/ capture-ms stored-bytes)
                                       (:calibration-alpha manager)))))
                  publish-task
                  (fn []
                    (try
                      (let [started (System/nanoTime)
                            result (publish-chunks!
                                    manager model-fingerprint stored)]
                        (swap! phase-timings assoc
                               :publication-ms
                               (/ (- (System/nanoTime) started) 1.0e6))
                        (.complete published result))
                      (catch Throwable error
                        (.completeExceptionally published error))))]
              (try
                (.execute ^ExecutorService (:publish-executor manager)
                          ^Runnable publish-task)
                (.complete captured (mapv local-node stored))
                (catch RejectedExecutionException error
                  (.complete captured (mapv local-node stored))
                  (.completeExceptionally published error))))
            (catch Throwable error
              (reject! error))))]
    (cond
      (and estimated-staging-bytes
           (> (long estimated-staging-bytes)
              (long (:max-chunk-staging-bytes manager))))
      (let [error (ex-info "Checkpoint chunk exceeds the host staging budget"
                           {:estimated-staging-bytes (long estimated-staging-bytes)
                            :max-chunk-staging-bytes
                            (:max-chunk-staging-bytes manager)})]
        (swap! (:metrics manager)
               #(-> %
                    (update :capture-rejected inc)
                    (update :capture-byte-rejected inc)))
        (reject! error)
        (assoc ticket :accepted? false :rejection :staging-byte-budget))

      (.get ^AtomicBoolean (:closed? manager))
      (let [error (RejectedExecutionException. "Continuation manager is closed")]
        (swap! (:metrics manager) update :capture-rejected inc)
        (reject! error)
        (assoc ticket :accepted? false))
      :else
      (try
        (.execute ^ExecutorService (:capture-executor manager) ^Runnable capture-task)
        (swap! (:metrics manager) update :capture-accepted inc)
        ticket
        (catch RejectedExecutionException error
          (swap! (:metrics manager) update :capture-rejected inc)
          (reject! error)
          (assoc ticket :accepted? false)))))))

(defn checkpoint-gpu-chunks-async!
  "Capture only missing contiguous GPU chunks on bounded background workers.

  The call itself never performs GPU, storage, or Datahike work. Capture performs
  the unavoidable device-to-host copy and durable local Konserve writes; the
  second worker publishes one Datahike transaction. Queue saturation rejects the
  cache operation without slowing inference."
  [^Manager manager state]
  (submit-chunk-checkpoint!
   manager
   (fn []
     {:model-fingerprint (:continuation/model-fingerprint state)
      :plan (:chunks (chunk/continuation-plan state (:chunk-size manager)))
      :export #(continuation-gpu/export-gpu-chunks state %)})))

(defn- complete-export-after-error!
  [pool export error]
  (try
    (page-pool/complete-export-chunk! pool export)
    (catch Throwable completion-error
      (.addSuppressed ^Throwable error completion-error)))
  (throw error))

(defn- await-paged-export!
  [pool continuation-id model-fingerprint descriptor]
  (let [export (page-pool/submit-export-chunk!
                pool continuation-id model-fingerprint descriptor)]
    (loop []
      (if (try
            (page-pool/chunk-export-complete? pool export)
            (catch Throwable error
              (complete-export-after-error! pool export error)))
        (page-pool/complete-export-chunk! pool export)
        (do
          (try
            (Thread/sleep 1)
            (catch InterruptedException error
              (complete-export-after-error! pool export error)))
          (recur))))))

(defn checkpoint-paged-chunks-async!
  "Capture a resident paged route through the bounded durable chunk pipeline.

  `tokens` contains the exact token history for the route. The route's current
  logical token count defines the processed prefix; a later pending token may be
  present in `tokens`. The accepted capture worker submits retained device-to-host
  downloads and never awaits an incomplete event on the decoder thread. Physical
  copy/compute overlap remains backend-dependent: Raster reports an independent
  OpenCL transfer queue while current Level Zero shared-memory downloads complete
  inline. Each
  payload is written through the local/tiered Konserve store before the next chunk
  is allocated, bounding host staging to one chunk. Datahike is published only
  after durability. Queue saturation or a chunk larger than
  `:max-chunk-staging-bytes` rejects the optional checkpoint immediately. The
  returned ticket also contains a `:phase-timings` atom populated by the capture
  and publication workers, so callers can distinguish catalog, transfer,
  persistence, mmap preparation, and publication costs without instrumenting
  private internals."
  [^Manager manager pool continuation-id model-fingerprint tokens]
  (let [tokens (vec tokens)
        resident-route (page-pool/route pool continuation-id)
        estimated-staging-bytes
        (when resident-route
          (page-pool/chunk-payload-bytes
           pool (min (long (:token-count resident-route))
                     (long (:chunk-size manager)))))]
    (submit-chunk-checkpoint!
     manager
     (fn []
       (let [resident-route
             (or (page-pool/route pool continuation-id)
                 (throw (ex-info "Cannot checkpoint a nonresident paged continuation"
                                 {:continuation-id continuation-id})))
             processed (long (:token-count resident-route))]
         {:model-fingerprint model-fingerprint
          :plan (chunk/plan tokens processed (:chunk-size manager))
          :export (fn [descriptor]
                    (if (:groups (:layout pool))
                      (mapv #(await-paged-export!
                              pool continuation-id model-fingerprint
                              (assoc descriptor :chunk/group (:id %)))
                            (:groups (:layout pool)))
                      [(await-paged-export!
                        pool continuation-id model-fingerprint descriptor)]))}))
     {:estimated-staging-bytes estimated-staging-bytes})))

(defn lookup-chunk-prefix
  "Return planned chunks, present entries, and the longest reusable KV prefix.

  Options accept `:maximum-cached-token-count`. Descriptors ending beyond that
  exact policy boundary are not queried or restored, even when a longer catalog
  prefix exists."
  ([^Manager manager model-fingerprint tokens]
   (lookup-chunk-prefix manager model-fingerprint tokens {}))
  ([^Manager manager model-fingerprint tokens
    {:keys [maximum-cached-token-count]}]
   (let [tokens (vec tokens)
         processed (max 0 (dec (count tokens)))
         maximum (long (or maximum-cached-token-count processed))
         _ (when-not (<= 0 maximum processed)
             (throw (ex-info "Maximum cached token count is outside the prompt"
                             {:maximum-cached-token-count maximum
                              :processed-token-count processed})))
         descriptors (->> (chunk/plan tokens processed (:chunk-size manager))
                          (take-while #(<= (+ (:chunk/start %)
                                              (:chunk/token-count %))
                                           maximum))
                          vec)
         entries (catalog/lookup-chunks
                  @(:connection manager) model-fingerprint
                  (mapv :chunk/prefix-hash descriptors))
         matched (catalog/longest-prefix descriptors entries)
         cached-token-count (reduce + 0 (map :kv/token-count matched))]
     (swap! (:metrics manager)
            (fn [metrics]
              (-> metrics
                  (update :prefix-lookups inc)
                  (update :requested-tokens + processed)
                  (update :cached-tokens + cached-token-count)
                  (update (cond (= cached-token-count processed) :full-hits
                                (zero? cached-token-count) :misses
                                :else :partial-hits) inc))))
     {:descriptors descriptors
      :entries entries
      :matched matched
      :cached-token-count cached-token-count})))

(defn- restore-record
  "Return the layout record for restoring `model-fingerprint` into
  `durable-layout`, the layout the restore target stores.

  A recorded layout must equal the target's; a catalog written before layout
  records existed falls back to the target's layout."
  [^Manager manager model-fingerprint durable-layout]
  (let [recorded (or (get @(:layout-records manager) model-fingerprint)
                     ;; A recorded layout never changes, so it is cached per
                     ;; fingerprint once found.
                     (when-let [record (catalog/lookup-layout @(:connection manager)
                                                              model-fingerprint)]
                       (swap! (:layout-records manager) assoc model-fingerprint record)
                       record))]
    (when (and recorded (not= (:layout recorded) durable-layout))
      (throw (ex-info "The fingerprint's recorded durable layout differs from the restore target"
                      {:model-fingerprint model-fingerprint
                       :recorded (:layout recorded)
                       :target durable-layout})))
    (when (and (nil? recorded) (get-in durable-layout [:attention-state :groups]))
      ;; Only single-group catalogs predate layout records, so a multi-group
      ;; chain without one cannot be checked against its target.
      (throw (ex-info "A multi-group restore requires the fingerprint's recorded layout"
                      {:model-fingerprint model-fingerprint})))
    (or recorded
        (catalog/layout-record model-fingerprint durable-layout
                               (:numerical-contract manager)))))

(defn- paged-load-plan
  "Return the certified manifest's load plan for the matched chain."
  [^Manager manager pool model-fingerprint matched]
  (if (empty? matched)
    {:parts [] :window-floors {}}
    (manifest/load-plan
     (manifest/boundary-manifest
      (restore-record manager model-fingerprint (page-pool/durable-layout pool))
      model-fingerprint matched))))

(defn- part-descriptor
  [pool part]
  {:chunk/start (:start part)
   :chunk/token-count (:token-count part)
   :chunk/group (:group part)
   :chunk/layout (page-pool/durable-layout pool)})

(defn restore-gpu-prefix
  "Restore the longest cached prompt prefix and compute only its missing suffix.

  Each chunk payload is mmaped and uploaded within a separate scope, bounding
  address-space and page residency. Returns cache statistics with a ready GPU
  continuation under `:continuation`."
  [^Manager manager dstate model-fingerprint tokens]
  (when-let [window (attention-state/min-window (:model dstate))]
    ;; Contiguous decode refuses spans beyond a sliding window; refuse the
    ;; restore before loading rows it could never use.
    (when (> (count tokens) (long window))
      (throw (ex-info (str "Contiguous GPU decode does not apply sliding windows; "
                           "use :cache-mode :paged beyond the window")
                      {:tokens (count tokens) :window window}))))
  (let [{:keys [matched cached-token-count] :as lookup-result}
        (lookup-chunk-prefix manager model-fingerprint tokens)]
    ;; Contiguous decode attends to the whole cached span, so every part of
    ;; every chunk is restored regardless of retention group.
    (doseq [entry matched
            part (parts/entry-parts entry)]
      (chunk-store/with-mmap-payload
       (:chunk-store manager) (:store-key part)
       (fn [payload]
         (when-not (and (= :float32 (:element-type payload))
                        (= :little-endian (:byte-order payload)))
           (throw (ex-info "Stored KV chunk is not a little-endian FP32 payload"
                           {:store-key (:store-key part)
                            :element-type (:element-type payload)
                            :byte-order (:byte-order payload)})))
         (continuation-gpu/upload-gpu-chunk!
          dstate (assoc entry :chunk/group (:group part)) (:segment payload)))))
    (swap! (:metrics manager)
           (fn [metrics]
             (-> metrics
                 (update :restored-chunks + (count matched))
                 (update :restored-bytes +
                         (reduce + 0 (for [entry matched
                                           part (parts/entry-parts entry)]
                                       (long (:bytes part))))))))
    (assoc lookup-result
           :continuation
           (continuation-gpu/resume-prompt-from-prefix
            dstate model-fingerprint tokens cached-token-count))))

(defn restore-paged-prefix!
  "Restore the longest cached prompt prefix into a worker-local device page pool.

  A new route named by `continuation-id` is allocated for the matched prefix.
  Each Hasch-addressed Konserve payload is mmaped only for its synchronous
  scatter into arbitrary physical pages. On any failure the partial route and
  all of its page references are released. Returns the ordinary lookup result
  with `:resident-route` and `:restore-phase-timings`; uncached prompt suffix
  evaluation remains the caller's model-execution responsibility.

  The optional `opts` arity enables `:admit?`. A cost-aware admission then evicts
  only durable, unpinned, unleased routes and installs `:policy` on the restored
  route. `:protected-continuation-ids` excludes active working-set routes.
  `:capacity-reservation` claims pages previously held for prompt restoration
  and subsequent generation; it is mutually exclusive with `:admit?` because
  admission must occur before the reservation is created.
  `:maximum-cached-token-count` enforces a shorter prefix selected by routing
  policy instead of implicitly loading a longer catalog/object-store tail."
  ([^Manager manager pool continuation-id model-fingerprint tokens]
   (restore-paged-prefix! manager pool continuation-id model-fingerprint tokens
                          {:policy {}}))
  ([^Manager manager pool continuation-id model-fingerprint tokens
    {:keys [admit? policy protected-continuation-ids capacity-reservation
            maximum-cached-token-count]
     :or {admit? false policy {:durable? true}
          protected-continuation-ids #{}}}]
   (when (and admit? capacity-reservation)
     (throw (ex-info "Paged restore cannot admit an existing reservation"
                     {:continuation-id continuation-id})))
   (let [restore-started (System/nanoTime)
         lookup-started restore-started
         {:keys [matched cached-token-count] :as lookup-result}
         (lookup-chunk-prefix
          manager model-fingerprint tokens
          {:maximum-cached-token-count maximum-cached-token-count})
         lookup-finished (System/nanoTime)
         ;; The plan is computed before allocation: a layout mismatch allocates
         ;; nothing, and the route is created with its window floors.
         {load-parts :parts window-floors :window-floors}
         (paged-load-plan manager pool model-fingerprint matched)
         allocation-started (System/nanoTime)
         admission (when admit?
                     (residency/admit-route!
                      pool continuation-id cached-token-count
                      {:policy policy
                       :window-floors window-floors
                       :protected-continuation-ids protected-continuation-ids}))
         _ (when (and admission (not (:admissible? admission)))
             (throw (ex-info "Cached prefix cannot be admitted to the GPU page pool"
                             (dissoc admission :resident-route))))
         resident-route (if admission
                          (:resident-route admission)
                          (page-pool/allocate-route!
                           pool continuation-id cached-token-count
                           {:policy policy
                            :window-floors window-floors
                            :capacity-reservation capacity-reservation}))
         allocation-finished (System/nanoTime)
         mapping-lifecycle-ns (volatile! 0)
         gpu-restore-ns (volatile! 0)]
     (try
       (do
         (doseq [part load-parts]
           (let [mapping-started (System/nanoTime)
                 transfer-ns (volatile! 0)]
             (chunk-store/with-mmap-payload
              (:chunk-store manager) (:store-key part)
              (fn [payload]
                (when-not (and (= :int16 (:element-type payload))
                               (= :little-endian (:byte-order payload)))
                  (throw (ex-info "Stored KV chunk is not a little-endian FP16 carrier payload"
                                  {:store-key (:store-key part)
                                   :element-type (:element-type payload)
                                   :byte-order (:byte-order payload)})))
                (let [started (System/nanoTime)]
                  (page-pool/restore-chunk!
                   pool continuation-id (part-descriptor pool part) (:segment payload))
                  (vreset! transfer-ns (- (System/nanoTime) started)))))
             (let [lifecycle-ns (- (System/nanoTime) mapping-started)]
               (vswap! gpu-restore-ns + @transfer-ns)
               (vswap! mapping-lifecycle-ns + (- lifecycle-ns @transfer-ns)))))
         (swap! (:metrics manager)
                (fn [metrics]
                  (-> metrics
                      (update :restored-chunks + (count matched))
                      (update :restored-bytes +
                              (reduce + 0 (map #(long (:bytes %)) load-parts)))))))
       (cond-> (assoc lookup-result
                      :resident-route resident-route
                      :restore-phase-timings
                      {:lookup-ms (/ (- lookup-finished lookup-started) 1.0e6)
                       :route-allocation-ms
                       (/ (- allocation-finished allocation-started) 1.0e6)
                       :mapping-lifecycle-ms (/ @mapping-lifecycle-ns 1.0e6)
                       :gpu-restore-ms (/ @gpu-restore-ns 1.0e6)
                       :total-ms (/ (- (System/nanoTime) restore-started) 1.0e6)
                       :chunks (count matched)})
         admission (assoc :admission (dissoc admission :resident-route)))
       (catch Throwable error
         (page-pool/release-route! pool continuation-id)
         (throw error))))))

(defn- cancellation-error
  [continuation-id stage]
  (CancellationException.
   (str "Paged continuation restore was cancelled at " (name stage)
        " boundary for " continuation-id)))

(defn- release-storage-after-error!
  [provider event error]
  (try
    (content/release-storage-event! provider event)
    (catch Throwable release-error
      (.addSuppressed ^Throwable error release-error)))
  (throw error))

(defn- complete-transfer-after-error!
  [pool transfer error]
  (try
    (page-pool/complete-restore-chunk! pool transfer)
    (catch Throwable completion-error
      (.addSuppressed ^Throwable error completion-error)))
  (throw error))

(defn- await-storage-boundary!
  [provider event continuation-id cancelled? poll-ms]
  (loop [cancel-requested? (boolean (cancelled?))]
    (if (try
          (content/storage-event-complete? provider event)
          (catch Throwable error
            (release-storage-after-error! provider event error)))
      (let [value (try
                    (content/await-storage-event! provider event)
                    (finally
                      (content/release-storage-event! provider event)))]
        (when (or cancel-requested? (cancelled?))
          (throw (cancellation-error continuation-id :storage)))
        value)
      (do
        (Thread/sleep (long poll-ms))
        (recur (or cancel-requested? (boolean (cancelled?))))))))

(defn- await-transfer-boundary!
  [pool transfer continuation-id cancelled? poll-ms]
  (loop [cancel-requested? (boolean (cancelled?))]
    (if (try
          (page-pool/chunk-transfer-complete? pool transfer)
          (catch Throwable error
            (complete-transfer-after-error! pool transfer error)))
      (do
        (page-pool/complete-restore-chunk! pool transfer)
        (when (or cancel-requested? (cancelled?))
          (throw (cancellation-error continuation-id :transfer))))
      (do
        (Thread/sleep (long poll-ms))
        (recur (or cancel-requested? (boolean (cancelled?))))))))

(defn restore-paged-prefix-overlapped!
  "Restore a cached paged prefix while unrelated decoder lanes keep running.

  Storage localization and GPU transfers expose nonblocking completion polls.
  Each mmap lease transfers to Raster with its upload event and closes only at
  device completion. Cancellation admits no new chunk but preserves the active
  storage/transfer boundary before releasing the partial route.

  Options accept `:policy`, `:capacity-reservation`,
  `:maximum-cached-token-count`, a zero-argument `:cancelled?` predicate, and a
  positive `:poll-ms` (default 1). The caller must run this function outside the
  decoder-owning loop, for example with `paged-runtime/run-background-operation!`."
  ([^Manager manager pool continuation-id model-fingerprint tokens]
   (restore-paged-prefix-overlapped!
    manager pool continuation-id model-fingerprint tokens {}))
  ([^Manager manager pool continuation-id model-fingerprint tokens
    {:keys [policy capacity-reservation maximum-cached-token-count
            cancelled? poll-ms]
     :or {policy {:durable? true}
          cancelled? (constantly false)
          poll-ms 1}}]
   (when-not (ifn? cancelled?)
     (throw (ex-info "Paged restore cancellation predicate must be callable" {})))
   (when-not (and (integer? poll-ms) (pos? poll-ms))
     (throw (ex-info "Paged restore polling interval must be positive"
                     {:poll-ms poll-ms})))
   (when (cancelled?)
     (throw (cancellation-error continuation-id :admission)))
   (let [{:keys [matched cached-token-count] :as lookup-result}
         (lookup-chunk-prefix
          manager model-fingerprint tokens
          {:maximum-cached-token-count maximum-cached-token-count})
         {load-parts :parts window-floors :window-floors}
         (paged-load-plan manager pool model-fingerprint matched)
         resident-route
         (page-pool/allocate-route!
          pool continuation-id cached-token-count
          {:policy policy :window-floors window-floors
           :capacity-reservation capacity-reservation})
         provider (:content-provider manager)]
     (try
       (do
       (doseq [part load-parts]
         (when (cancelled?)
           (throw (cancellation-error continuation-id :chunk)))
         (let [address (content-provider/content-address (:store-key part))
               localization (content/submit-localization! provider address)
               _ (await-storage-boundary!
                  provider localization continuation-id cancelled? poll-ms)
               lease (content/open-local-content! provider address)
               submitted? (volatile! false)]
           (try
             (let [placement (:placement lease)
                   payload (merge (:attributes placement)
                                  {:segment (content/lease-segment lease)})]
               (when-not (and (= :int16 (:element-type payload))
                              (= :little-endian (:byte-order payload)))
                 (throw (ex-info
                         "Stored KV chunk is not a little-endian FP16 carrier payload"
                         {:store-key (:store-key part)
                          :element-type (:element-type payload)
                          :byte-order (:byte-order payload)})))
               (let [transfer
                     (page-pool/submit-restore-chunk!
                      pool continuation-id (part-descriptor pool part)
                      (:segment payload) [lease])]
                 (vreset! submitted? true)
                 (await-transfer-boundary!
                  pool transfer continuation-id cancelled? poll-ms)))
             (finally
               (when-not @submitted?
                 (.close ^AutoCloseable lease))))))
       ;; Chunks and bytes are counted together, only for a completed restore.
       (swap! (:metrics manager)
              (fn [metrics]
                (-> metrics
                    (update :restored-chunks + (count matched))
                    (update :restored-bytes +
                            (reduce + 0 (map #(long (:bytes %)) load-parts)))))))
       (when (cancelled?)
         (throw (cancellation-error continuation-id :completion)))
       (assoc lookup-result :resident-route resident-route)
       (catch Throwable error
         (page-pool/release-route! pool continuation-id)
         (throw error))))))

(defn- publish-file-with!
  [^Manager manager snapshot writer]
  (let [directory (:directory manager)
        temporary (Files/createTempFile directory "pending-" ".rstrkv"
                                        (make-array java.nio.file.attribute.FileAttribute 0))]
    (try
      (let [stored (writer snapshot (.toFile temporary))
            final-path (.resolve directory (str (:content-id stored) ".rstrkv"))]
        (try
          (Files/move temporary final-path
                      (into-array StandardCopyOption
                                  [StandardCopyOption/REPLACE_EXISTING
                                   StandardCopyOption/ATOMIC_MOVE]))
          (catch java.nio.file.AtomicMoveNotSupportedException _
            (Files/move temporary final-path
                        (into-array StandardCopyOption [StandardCopyOption/REPLACE_EXISTING]))))
        (assoc stored :path (.toString final-path)))
      (finally
        (Files/deleteIfExists temporary)))))

(defn- publish-file!
  [^Manager manager snapshot]
  (publish-file-with! manager snapshot store/write-snapshot!))

(defn- catalog-entry!
  [^Manager manager snapshot stored]
  (let [tokens (:continuation/tokens snapshot)
        entry {:kv/id (random-uuid)
               :kv/model-fingerprint (:continuation/model-fingerprint snapshot)
               :kv/prefix-hash (catalog/token-prefix-hash tokens)
               :kv/processed-count (:continuation/processed-count snapshot)
               :kv/pending-token (:continuation/pending-token snapshot)
               :kv/logical-token-count (long (count tokens))}]
    (catalog/put! (:connection manager) entry stored)
    (merge entry {:kv/blob (:content-id stored) :kv/path (:path stored)
                  :kv/bytes (:bytes stored)})))

(defn checkpoint-cpu!
  "Persist a CPU continuation and publish its exact-prefix catalog entry.

  The immutable file is fully written and renamed before the Datahike transaction.
  A crash in that gap can leave an unreferenced file; an age-based sweep may remove
  such files without risking a referenced object."
  [^Manager manager state]
  (let [snapshot (continuation/export-cpu state)
        stored (publish-file! manager snapshot)]
    (catalog-entry! manager snapshot stored)))

(defn checkpoint-gpu!
  "Persist a GPU continuation directly into a writable mmap and catalog it.

  Raster synchronously downloads the occupied K/V prefix into mapped tensor
  slabs; no intermediate JVM tensor arrays are allocated. The immutable file is
  renamed before its Datahike entry becomes visible."
  [^Manager manager state]
  (let [snapshot (continuation-gpu/export-gpu-metadata state)
        stored (publish-file-with!
                manager snapshot
                (fn [metadata file]
                  (store/write-snapshot-with!
                   metadata file #(continuation-gpu/export-gpu-into! state %))))]
    (catalog-entry! manager snapshot stored)))

(defn checkpoint-gpu-async!
  "Capture and publish a GPU continuation without blocking the inference thread.

  The returned ticket is
  `{:accepted? boolean :captured CompletableFuture :published CompletableFuture}`.
  Capture copies only the immutable occupied prefix into a local mmap. Publication
  performs the Datahike transaction on a separate worker. The GPU session must
  remain valid until `:captured` completes, but need not remain valid while
  `:published` is pending.

  If the bounded capture queue is full, `:accepted?` is false and both futures
  complete exceptionally; inference is never used to drain cache work."
  [^Manager manager state]
  (let [captured (CompletableFuture.)
        published (CompletableFuture.)
        ticket {:accepted? true :captured captured :published published}
        reject! (fn [error]
                  (.completeExceptionally captured error)
                  (.completeExceptionally published error))
        capture-task
        (fn []
          (try
            (let [snapshot (continuation-gpu/export-gpu-metadata state)
                  stored (publish-file-with!
                          manager snapshot
                          (fn [metadata file]
                            (store/write-snapshot-with!
                             metadata file
                             #(continuation-gpu/export-gpu-into! state %))))
                  capture-result {:snapshot snapshot :stored stored}
                  publish-task
                  (fn []
                    (try
                      (.complete published (catalog-entry! manager snapshot stored))
                      (catch Throwable error
                        (.completeExceptionally published error))))]
              (try
                (.execute ^ExecutorService (:publish-executor manager)
                          ^Runnable publish-task)
                (.complete captured capture-result)
                (catch RejectedExecutionException error
                  (.complete captured capture-result)
                  (.completeExceptionally published error))))
            (catch Throwable error
              (reject! error))))]
    (if (.get ^AtomicBoolean (:closed? manager))
      (let [error (RejectedExecutionException. "Continuation manager is closed")]
        (swap! (:metrics manager) update :capture-rejected inc)
        (reject! error)
        (assoc ticket :accepted? false))
      (try
        (.execute ^ExecutorService (:capture-executor manager) ^Runnable capture-task)
        (swap! (:metrics manager) update :capture-accepted inc)
        ticket
        (catch RejectedExecutionException error
          (swap! (:metrics manager) update :capture-rejected inc)
          (reject! error)
          (assoc ticket :accepted? false))))))

(defn advance-gpu-with-checkpoints!
  "Advance GPU generation while opportunistically scheduling checkpoints.

  `opts` must contain positive `:checkpoint-every` tokens and may contain
  `:checkpoint-final?` and `:chunked?`. With `:chunked? true`, checkpoints reuse
  immutable hash-chain nodes and store only missing chunks. Otherwise the archival
  whole-prefix snapshot path is retained. Saturated queues return rejected tickets
  and generation continues.

  Returns `{:continuation state :tokens generated :checkpoints tickets}`."
  [^Manager manager state n {:keys [checkpoint-every checkpoint-final? chunked?]}]
  (let [interval (long checkpoint-every)
        checkpoint! (if chunked? checkpoint-gpu-chunks-async!
                         checkpoint-gpu-async!)]
    (when-not (pos? interval)
      (throw (ex-info ":checkpoint-every must be positive"
                      {:checkpoint-every checkpoint-every})))
    (let [processed (long (:continuation/processed-count state))
          first-boundary (* (inc (quot processed interval)) interval)]
      (loop [state state
             remaining (long n)
             generated []
             checkpoints []
             next-boundary first-boundary
             last-checkpoint nil]
        (if (zero? remaining)
          (let [processed (long (:continuation/processed-count state))
                final? (and checkpoint-final? (not= processed last-checkpoint))]
            {:continuation state
             :tokens generated
             :checkpoints (cond-> checkpoints
                            final? (conj (checkpoint! manager state)))})
          (let [[next-state token] (continuation-gpu/step-gpu state)
                processed (long (:continuation/processed-count next-state))
                checkpoint? (>= processed next-boundary)]
            (recur next-state
                   (dec remaining)
                   (conj generated token)
                   (cond-> checkpoints
                     checkpoint? (conj (checkpoint! manager next-state)))
                   (if checkpoint? (+ next-boundary interval) next-boundary)
                   (if checkpoint? processed last-checkpoint))))))))

(defn lookup
  "Look up an exact logical token prefix for a model."
  [^Manager manager model-fingerprint tokens]
  (catalog/lookup @(:connection manager) model-fingerprint
                  (catalog/token-prefix-hash tokens)))

(defn restore-cpu
  "Map a catalog entry, materialize its CPU tensors and restore runtime capacity."
  [entry model opts]
  (with-open [mapped (store/open-snapshot (:kv/path entry))]
    (continuation/restore-cpu model (store/materialize (:snapshot mapped)) opts)))

(defn restore-gpu
  "Map a catalog entry and restore it directly into an existing GPU decoder.

  Raster's upload is synchronous, so the mapping can close before the returned
  continuation is advanced. No intermediate JVM tensor arrays are allocated."
  [entry dstate opts]
  (with-open [mapped (store/open-snapshot (:kv/path entry))]
    (continuation-gpu/restore-gpu dstate (:snapshot mapped) opts)))

(defn evict!
  "Retract a catalog entry and delete its external immutable blob."
  [^Manager manager entry]
  (catalog/retract! (:connection manager) (:kv/id entry))
  (Files/deleteIfExists (.toPath (java.io.File. ^String (:kv/path entry)))))
