(ns pretrained.continuation.content-provider
  "Raster numerical-content provider for immutable Konserve KV chunks."
  (:require [pretrained.continuation.chunk-store :as chunk-store]
            [raster.compiler.ir.numerical-state :as numerical-state]
            [raster.runtime.numerical-content :as content])
  (:import [java.io Closeable]
           [java.lang AutoCloseable]
           [java.lang.foreign MemorySegment]
           [java.util UUID]
           [java.util.concurrent ArrayBlockingQueue CompletableFuture ExecutionException
            ExecutorService RejectedExecutionException ThreadFactory ThreadPoolExecutor
            ThreadPoolExecutor$AbortPolicy TimeUnit]
           [java.util.concurrent.atomic AtomicBoolean AtomicLong]))

(def ^:private address-algorithm :hasch/uuid)

(defn content-address
  "Return Raster's storage-neutral address for a Hasch-addressed chunk key."
  [store-key]
  (when-not (instance? UUID store-key)
    (throw (ex-info "Konserve chunk keys must be Hasch UUIDs"
                    {:store-key store-key})))
  (numerical-state/content-address address-algorithm (str store-key)))

(defn- address-store-key
  [address]
  (when-not (and (numerical-state/content-address? address)
                 (= address-algorithm (:algorithm address)))
    (throw (ex-info "Numerical content is not a Hasch-addressed KV chunk"
                    {:content address :expected-algorithm address-algorithm})))
  (try
    (UUID/fromString (:digest address))
    (catch IllegalArgumentException error
      (throw (ex-info "Numerical content has an invalid Hasch UUID digest"
                      {:content address} error)))))

(defn- daemon-thread-factory
  []
  (let [counter (AtomicLong.)]
    (reify ThreadFactory
      (newThread [_ runnable]
        (doto (Thread. runnable
                       (str "pretrained-content-localize-"
                            (.incrementAndGet counter)))
          (.setDaemon true)
          (.setPriority Thread/MIN_PRIORITY))))))

(defn- bounded-executor
  [concurrency]
  (ThreadPoolExecutor.
   (int concurrency) (int concurrency) 0 TimeUnit/MILLISECONDS
   (ArrayBlockingQueue. (int concurrency))
   (daemon-thread-factory)
   (ThreadPoolExecutor$AbortPolicy.)))

(defn- submit-event!
  [provider operation task]
  (when (.get ^AtomicBoolean (:closed? provider))
    (swap! (:metrics provider) update :localizations-rejected inc)
    (throw (RejectedExecutionException. "Konserve content provider is closed")))
  (let [description (:descriptor provider)
        event (content/storage-event
               {:provider-id (:id description)
                :id (random-uuid)
                :operation operation})
        future (CompletableFuture.)
        submitted-ns (System/nanoTime)]
    (swap! (:events provider) assoc (:id event)
           {:event event :future future :submitted-ns submitted-ns})
    (try
      (.execute
       ^ExecutorService (:executor provider)
       ^Runnable
       (reify Runnable
         (run [_]
           (try
             (let [value (task)
                   completed-ns (System/nanoTime)
                   measurement
                   {:timing-source :host-wall
                    :operation operation
                    :bytes (long (or (get-in value
                                             [:attributes :transferred-bytes])
                                     0))
                    :elapsed-ns (- completed-ns submitted-ns)}]
               (swap! (:events provider) assoc-in
                      [(:id event) :measurement] measurement)
               (swap! (:metrics provider)
                      (fn [metrics]
                        (-> metrics
                            (update :localizations-completed inc)
                            (update :localized-bytes + (:bytes measurement))
                            (update :localization-elapsed-ns +
                                    (:elapsed-ns measurement)))))
               (.complete future value))
             (catch Throwable error
               (.completeExceptionally future error))))))
      event
      (catch RejectedExecutionException error
        (swap! (:events provider) dissoc (:id event))
        (swap! (:metrics provider) update :localizations-rejected inc)
        (.completeExceptionally future error)
        (throw error)))))

(defn- event-entry
  [provider event]
  (or (get @(:events provider) (:id event))
      (throw (ex-info "Konserve storage event is no longer owned"
                      {:event event}))))

(defn- await-future
  [^CompletableFuture future]
  (try
    (.get future)
    (catch ExecutionException error
      (throw (.getCause error)))))

(defn- localize!
  [provider address]
  (let [store-key (address-store-key address)
        local-store (:local-store provider)
        local? (chunk-store/stored? local-store store-key)]
    (when-not local?
      (let [missing (Object.)
            chunk (chunk-store/read-chunk (:read-store provider)
                                          store-key missing)]
        (when (identical? missing chunk)
          (throw (ex-info "Numerical content is absent from every Konserve tier"
                          {:content address :store-key store-key})))
        (let [actual-key (chunk-store/content-id chunk)]
          (when-not (= store-key actual-key)
            (throw (ex-info "Localized chunk differs from its content address"
                            {:expected store-key :actual actual-key}))))
        ;; Frontend-first tiering warms asynchronously. An explicit immutable put
        ;; makes localization completion a real local-readiness boundary.
        (when-not (chunk-store/stored? local-store store-key)
          (chunk-store/put! local-store chunk))))
    (let [{:keys [bytes path]} (chunk-store/describe local-store store-key)
          payload (chunk-store/prepare-mmap-payload! local-store store-key)]
      (content/content-placement
       {:provider-id (get-in provider [:descriptor :id])
        :tier-id :local
        :content address
        :attributes {:store-key store-key :bytes bytes :path path
                     :payload-file-offset (:file-offset payload)
                     :payload-byte-size (:byte-size payload)
                     :payload-element-type (:element-type payload)
                     :source-tier (if local? :local :authoritative)
                     :transferred-bytes (if local? 0 bytes)}}))))

(defrecord KonserveContentProvider
    [descriptor local-store read-store executor events metrics closed?]
  content/ContentProvider
  (-provider-descriptor [_] descriptor)
  (-submit-promotion! [_ _ _ _]
    (throw (UnsupportedOperationException.
            "This provider localizes chunks; manager durability receipts publish them")))
  (-submit-localization! [this address _]
    (submit-event! this :localize #(localize! this address)))
  (-open-local-content! [_ address _]
    (when (.get ^AtomicBoolean closed?)
      (throw (RejectedExecutionException. "Konserve content provider is closed")))
    (let [store-key (address-store-key address)]
      (when-not (chunk-store/stored? local-store store-key)
        (throw (ex-info "Numerical content must be localized before it is opened"
                        {:content address :store-key store-key})))
      (let [[payload arena] (chunk-store/mmap-payload local-store store-key)
            segment (:segment payload)]
        (try
          (content/local-content-lease
           {:content address
            :placement
            (content/content-placement
             {:provider-id (:id descriptor)
              :tier-id :local
              :content address
              :attributes
              (merge {:store-key store-key}
                     (dissoc payload :segment))})
            :segment segment
            :byte-length (.byteSize ^MemorySegment segment)
            :release-fn #(.close ^AutoCloseable arena)})
          (catch Throwable error
            (.close ^AutoCloseable arena)
            (throw error))))))
  (-storage-event-complete? [this event]
    (.isDone ^CompletableFuture (:future (event-entry this event))))
  (-await-storage-event! [this event]
    (await-future (:future (event-entry this event))))
  (-storage-event-measurement [this event]
    (let [{:keys [future measurement]} (event-entry this event)]
      (when (and (.isDone ^CompletableFuture future)
                 (not (.isCompletedExceptionally ^CompletableFuture future)))
        measurement)))
  (-release-storage-event! [this event]
    (let [entry (event-entry this event)]
      (try
        (await-future (:future entry))
        (finally
          (swap! events dissoc (:id event))))))
  Closeable
  (close [_]
    (when (.compareAndSet ^AtomicBoolean closed? false true)
      (.shutdown ^ExecutorService executor)
      (when-not (.awaitTermination ^ExecutorService executor 60 TimeUnit/SECONDS)
        (.shutdownNow ^ExecutorService executor)
        (.awaitTermination ^ExecutorService executor 10 TimeUnit/SECONDS))
      (reset! events {}))))

(defn stats
  "Return current localization event and bounded queue counts."
  [provider]
  (merge @(:metrics provider)
         {:localization-events (count @(:events provider))
          :localization-queue-depth
          (.size (.getQueue ^ThreadPoolExecutor (:executor provider)))
          :closed? (.get ^AtomicBoolean (:closed? provider))}))

(defn open-provider
  "Open a bounded asynchronous localization provider over Konserve stores.

  `local-store` must support `konserve.mmap`. `read-store` may be a tiered
  frontend-first store and defaults to the local store. The caller owns both
  stores; the returned provider owns only its localization executor.
  Localization validates and prepares the scoped tensor mapping, so first-use
  Boring/FFM initialization completes before the placement becomes ready."
  ([local-store] (open-provider local-store local-store {}))
  ([local-store read-store] (open-provider local-store read-store {}))
  ([local-store read-store {:keys [max-concurrent-localizations]
                            :or {max-concurrent-localizations 2}}]
   (when-not (and (integer? max-concurrent-localizations)
                  (pos? max-concurrent-localizations))
     (throw (ex-info "Content localization concurrency must be positive"
                     {:max-concurrent-localizations
                      max-concurrent-localizations})))
   (let [local-tier
         (content/storage-tier
          {:id :local :kind :file :locality :node :durability :cached
           :capabilities #{:scoped-segment :mmap}})
         tiers
         (cond-> [local-tier]
           (not (identical? local-store read-store))
           (conj
            (content/storage-tier
             {:id :authoritative :kind :konserve :locality :shared
              :durability :durable :capabilities #{}})))
         descriptor
         (content/provider-description
          {:id (random-uuid)
           :tiers tiers
           :capabilities #{:localize :scoped-segment}
           :attributes {:implementation :konserve-boring}})]
     (->KonserveContentProvider
      descriptor local-store read-store
      (bounded-executor max-concurrent-localizations)
      (atom {})
      (atom {:localizations-completed 0 :localizations-rejected 0
             :localized-bytes 0 :localization-elapsed-ns 0})
      (AtomicBoolean. false)))))
