(ns pretrained.continuation-manager-test
  (:require [clojure.core.async :refer [go <! promise-chan put!]]
            [clojure.test :refer [deftest is]]
            [datahike.api :as d]
            [konserve.core :as k]
            [konserve.memory :refer [new-mem-store]]
            [konserve.protocols :as protocols]
            [pretrained.attention-state :as attention-state]
            [pretrained.continuation :as continuation]
            [pretrained.continuation.catalog :as catalog]
            [pretrained.continuation.gpu :as continuation-gpu]
            [pretrained.continuation.manager :as manager]
            [pretrained.continuation.page-pool :as page-pool]
            [pretrained.decoder-gpu :as decoder-gpu]
            [raster.gpu.core :as gpu])
  (:import [java.lang AutoCloseable]
           [java.lang.foreign MemorySegment]
           [java.nio.file Files]
           [java.util.concurrent CancellationException CountDownLatch TimeUnit]))

(defn- delayed-assoc-store
  [backend-store ^CountDownLatch entered release-write]
  (reify protocols/PEDNKeyValueStore
    (-exists? [_ key opts]
      (protocols/-exists? backend-store key opts))
    (-get-meta [_ key opts]
      (protocols/-get-meta backend-store key opts))
    (-get-in [_ key-vec not-found opts]
      (protocols/-get-in backend-store key-vec not-found opts))
    (-update-in [_ key-vec meta-up-fn up-fn opts]
      (protocols/-update-in backend-store key-vec meta-up-fn up-fn opts))
    (-assoc-in [_ key-vec meta-up-fn val opts]
      (go
        (.countDown entered)
        (<! release-write)
        (<! (protocols/-assoc-in backend-store key-vec meta-up-fn val opts))))
    (-dissoc [_ key opts]
      (protocols/-dissoc backend-store key opts))))

(defn- delete-directory!
  [directory]
  (doseq [file (reverse (file-seq (.toFile directory)))]
    (Files/deleteIfExists (.toPath file))))

(deftest caller-owned-connection-survives-manager-close
  (let [directory (Files/createTempDirectory
                   "pretrained-kv-external-connection-"
                   (make-array java.nio.file.attribute.FileAttribute 0))
        config {:store {:backend :memory :id (random-uuid)}
                :schema-flexibility :write :keep-history? false :value-caps :default}
        connection (catalog/ensure-database! config)
        cache (manager/open-manager nil directory {:connection connection})]
    (try
      (is (identical? connection (manager/connection cache)))
      (is (some? (manager/local-chunk-store cache)))
      (.close cache)
      (is (map? (d/transact connection [{:db/ident :fixture/external-connection}])))
      (finally
        (d/release connection)
        (d/delete-database config)
        (delete-directory! directory)))))

(deftest checkpoint-query-restore-and-evict
  (let [directory (Files/createTempDirectory "pretrained-kv-manager-"
                                               (make-array java.nio.file.attribute.FileAttribute 0))
        config {:store {:backend :memory :id (random-uuid)}
                :schema-flexibility :write :keep-history? false :value-caps :default}
        model {:n-layers 1 :n-kv 1 :head-dim 2}
        state {:continuation/backend :cpu
               :continuation/model model
               :continuation/model-fingerprint "fixture-v1"
               :continuation/layout (continuation/model-layout model)
               :continuation/max-position 8
               :continuation/processed-count 2
               :continuation/pending-token 3
               :continuation/tokens [1 2 3]
               :continuation/keys [(float-array [10 11 12 13 0 0 0 0 0 0 0 0 0 0 0 0])]
               :continuation/values [(float-array [20 21 22 23 0 0 0 0 0 0 0 0 0 0 0 0])]}
        cache (manager/open-manager config directory)]
    (try
      (let [entry (manager/checkpoint-cpu! cache state)
            found (manager/lookup cache "fixture-v1" [1 2 3])
            restored (manager/restore-cpu found model
                                          {:max-position 8 :model-fingerprint "fixture-v1"})]
        (is (= (:kv/id entry) (:kv/id found)))
        (is (= [10.0 11.0 12.0 13.0]
               (vec (take 4 (first (:continuation/keys restored))))))
        (is (true? (manager/evict! cache found)))
        (is (nil? (manager/lookup cache "fixture-v1" [1 2 3]))))
      (finally
        (.close cache)
        (d/delete-database config)
        (delete-directory! directory)))))

(deftest checkpoint-query-and-direct-gpu-restore
  (let [directory (Files/createTempDirectory "pretrained-kv-gpu-manager-"
                                               (make-array java.nio.file.attribute.FileAttribute 0))
        config {:store {:backend :memory :id (random-uuid)}
                :schema-flexibility :write :keep-history? false :value-caps :default}
        model {:n-layers 1 :n-kv 1 :head-dim 2}
        dstate {:model model :maxpos 8 :sess ::session}
        state {:continuation/backend :gpu
               :continuation/dstate dstate
               :continuation/model-fingerprint "fixture-gpu-v1"
               :continuation/layout (continuation/model-layout model)
               :continuation/max-position 8
               :continuation/processed-count 2
               :continuation/pending-token 3
               :continuation/tokens [1 2 3]}
        downloaded (atom nil)
        uploaded (atom nil)
        cache (manager/open-manager config directory)]
    (try
      (with-redefs [gpu/download-ranges!
                    (fn [_ entries]
                      (reset! downloaded entries)
                      (doseq [[key ^MemorySegment destination {:keys [elements]}] entries]
                        (let [values (float-array elements)]
                          (java.util.Arrays/fill values
                                                 (float (if (= :kc0 key) 1.0 2.0)))
                          (MemorySegment/copy (MemorySegment/ofArray values) 0
                                              destination 0 (* 4 elements))))
                      (mapv second entries))
                    gpu/upload-ranges!
                    (fn [_ entries]
                      (reset! uploaded
                              (mapv (fn [[key ^MemorySegment source {:keys [elements]}]]
                                      (let [values (float-array elements)]
                                        (MemorySegment/copy source 0
                                                            (MemorySegment/ofArray values) 0
                                                            (* 4 elements))
                                        [key (vec values)]))
                                    entries))
                      (mapv second entries))
                    decoder-gpu/prime-resident-token! (fn [state* _] state*)]
        (let [entry (manager/checkpoint-gpu! cache state)
              found (manager/lookup cache "fixture-gpu-v1" [1 2 3])
              restored (manager/restore-gpu found dstate
                                            {:model-fingerprint "fixture-gpu-v1"})]
          (is (= (:kv/id entry) (:kv/id found)))
          (is (every? #(instance? MemorySegment (second %)) @downloaded)
              "GPU export writes directly into mapped slabs")
          (is (= [[:kc0 [1.0 1.0 1.0 1.0]]
                  [:vc0 [2.0 2.0 2.0 2.0]]]
                 @uploaded))
          (is (= 2 (:continuation/processed-count restored)))))
      (finally
        (.close cache)
        (d/delete-database config)
        (delete-directory! directory)))))

(deftest chunked-konserve-mmap-restores-longest-prefix
  (let [directory (Files/createTempDirectory "pretrained-kv-chunk-manager-"
                                               (make-array java.nio.file.attribute.FileAttribute 0))
        config {:store {:backend :memory :id (random-uuid)}
                :schema-flexibility :write :keep-history? false :value-caps :default}
        model {:n-layers 1 :n-kv 1 :head-dim 2}
        cpu-state {:continuation/backend :cpu
                   :continuation/model model
                   :continuation/model-fingerprint "fixture-chunks-v1"
                   :continuation/layout (continuation/model-layout model)
                   :continuation/max-position 8
                   :continuation/processed-count 4
                   :continuation/pending-token 5
                   :continuation/tokens [1 2 3 4 5]
                   :continuation/keys [(float-array [10 11 20 21 30 31 40 41
                                                     0 0 0 0 0 0 0 0])]
                   :continuation/values [(float-array [50 51 60 61 70 71 80 81
                                                       0 0 0 0 0 0 0 0])]}
        dstate {:model model :maxpos 8 :sess ::session}
        uploads (atom [])
        decoded (atom [])
        primed (atom nil)
        cache (manager/open-manager config directory {:chunk-size 2})]
    (try
      (manager/checkpoint-cpu-chunks! cache cpu-state)
      (with-redefs [gpu/upload-ranges!
                    (fn [_ entries]
                      (swap! uploads into
                             (mapv (fn [[key ^MemorySegment source
                                        {:keys [src-element dst-element elements]}]]
                                     (let [values (float-array elements)]
                                       (MemorySegment/copy source (* 4 src-element)
                                                           (MemorySegment/ofArray values) 0
                                                           (* 4 elements))
                                       [key dst-element (vec values)]))
                                   entries))
                      (mapv second entries))
                    decoder-gpu/decode-token!
                    (fn [_ token position] (swap! decoded conj [token position]))
                    decoder-gpu/prime-resident-token!
                    (fn [state* token] (reset! primed token) state*)]
        (let [result (manager/restore-gpu-prefix
                      cache dstate "fixture-chunks-v1" [1 2 3 4 5 6 7])]
          (is (= 4 (:cached-token-count result)))
          (is (= 2 (count (:matched result))))
          (is (= [[:kc0 0 [10.0 11.0 20.0 21.0]]
                  [:vc0 0 [50.0 51.0 60.0 61.0]]
                  [:kc0 4 [30.0 31.0 40.0 41.0]]
                  [:vc0 4 [70.0 71.0 80.0 81.0]]]
                 @uploads))
          (is (= [[5 4] [6 5]] @decoded))
          (is (= 7 @primed))
          (is (= 6 (get-in result [:continuation :continuation/processed-count])))
          (is (= {:chunks-planned 2 :chunks-reused 0 :chunks-stored 2
                  :prefix-lookups 1 :full-hits 0 :partial-hits 1 :misses 0
                  :requested-tokens 6 :cached-tokens 4
                  :restored-chunks 2}
                 (select-keys (manager/stats cache)
                              [:chunks-planned :chunks-reused :chunks-stored
                               :prefix-lookups :full-hits :partial-hits :misses
                               :requested-tokens :cached-tokens
                               :restored-chunks])))))
      (finally
        (.close cache)
        (d/delete-database config)
        (delete-directory! directory)))))

(deftest chunked-konserve-mmap-scatters-into-resident-pages
  (let [directory (Files/createTempDirectory
                   "pretrained-kv-paged-manager-"
                   (make-array java.nio.file.attribute.FileAttribute 0))
        config {:store {:backend :memory :id (random-uuid)}
                :schema-flexibility :write :keep-history? false :value-caps :default}
        model {:n-layers 1 :n-kv 1 :head-dim 2}
        pool (page-pool/->DevicePagePool
              ::session (attention-state/layout model) 2 4 :half
              {[:key 0] :pool-k0, [:value 0] :pool-v0}
              (atom {:free (apply sorted-set (range 4))
                     :refcounts {}
                     :routes {}}))
        uploads (atom [])
        cache (manager/open-manager config directory {:chunk-size 2})]
    (try
      (page-pool/allocate-route! pool :source 4)
      (with-redefs [page-pool/submit-export-chunk!
                    (fn [_ _ model-fingerprint descriptor]
                      (let [values (if (zero? (:chunk/start descriptor))
                                     [10 11 20 21 50 51 60 61]
                                     [30 31 40 41 70 71 80 81])]
                        (merge descriptor
                               {:chunk/version 3
                                :chunk/model-fingerprint model-fingerprint
                                :chunk/layout
                                {:dtype :float16
                                 :byte-order :little-endian
                                 :attention-state
                                 (assoc (attention-state/layout model)
                                        :dtype :float16)}
                                :chunk/elements-per-slab 4
                                :chunk/payload
                                (short-array
                                 (map #(Float/floatToFloat16 (float %)) values))})))
                    page-pool/chunk-export-complete? (fn [& _] true)
                    page-pool/complete-export-chunk! (fn [_ export] export)
                    gpu/buffer-view
                    (fn [_ key opts] {:key key :opts opts})
                    gpu/submit-upload-ranges! (fn [_ entries] entries)
                    gpu/submit-upload-ranges-retained!
                    (fn [_ entries resources]
                      {:entries entries :resources resources})
                    gpu/event-complete? (fn [& _] true)
                    gpu/await-event!
                    (fn [_ event]
                      (let [entries (if (map? event) (:entries event) event)]
                        (swap! uploads into
                               (mapv (fn [[view ^MemorySegment source
                                          {:keys [src-element elements] :as spec}]]
                                       (let [values (short-array elements)]
                                         (MemorySegment/copy
                                          source (* 2 src-element)
                                          (MemorySegment/ofArray values) 0
                                          (* 2 elements))
                                         [(:key view)
                                          (quot (get-in view [:opts :byte-offset]) 2)
                                          (mapv #(Float/float16ToFloat %) values)
                                          spec]))
                                     entries))
                        (doseq [resource (:resources event)]
                          (.close ^AutoCloseable resource))
                        (mapv second entries)))
                    gpu/event-measurement
                    (fn [& _] {:direction :upload :timing-source :host-monotonic
                               :asynchronous? false :bytes 32 :commands 4
                               :elapsed-ns 40 :submit-host-ns 40 :host-wall-ns 40})
                    gpu/release-event! (fn [& _] nil)]
        (let [ticket (manager/checkpoint-paged-chunks-async!
                      cache pool :source "fixture-paged-v1" [1 2 3 4 5])
              _ (.get ^java.util.concurrent.CompletableFuture
                      (:published ticket) 5 TimeUnit/SECONDS)
              limited (manager/lookup-chunk-prefix
                       cache "fixture-paged-v1" [1 2 3 4 5 6 7]
                       {:maximum-cached-token-count 2})
              _ (page-pool/release-route! pool :source)
              capacity (page-pool/reserve-capacity! pool :request-a 7)
              result (manager/restore-paged-prefix!
                      cache pool :request-a "fixture-paged-v1"
                      [1 2 3 4 5 6 7]
                      {:capacity-reservation capacity
                       :policy {}})]
          (is (= 2 (:cached-token-count limited)))
          (is (= 1 (count (:matched limited))))
          (is (= 4 (:cached-token-count result)))
          (is (= 2 (get-in result [:restore-phase-timings :chunks])))
          (is (every? #(not (neg? (double %)))
                      (map (:restore-phase-timings result)
                           [:lookup-ms :route-allocation-ms
                            :mapping-lifecycle-ms :gpu-restore-ms :total-ms])))
          (is (= {:continuation-id :request-a
                  :pages [0 1]
                  :token-count 4
                  :start-position 0}
                 (:resident-route result)))
          (is (= [[:pool-k0 0 [10.0 11.0 20.0 21.0] {:src-element 0
                                                       :dst-element 0
                                                       :elements 4}]
                  [:pool-v0 0 [50.0 51.0 60.0 61.0] {:src-element 4
                                                       :dst-element 0
                                                       :elements 4}]
                  [:pool-k0 4 [30.0 31.0 40.0 41.0] {:src-element 0
                                                       :dst-element 0
                                                       :elements 4}]
                  [:pool-v0 4 [70.0 71.0 80.0 81.0] {:src-element 4
                                                       :dst-element 0
                                                       :elements 4}]]
                 @uploads))
          (is (= 2 (:reserved-pages (page-pool/stats pool))))
          (is (= 2 (:restored-chunks (manager/stats cache))))
          (is (page-pool/release-capacity! pool capacity))
          (is (page-pool/release-route! pool :request-a))
          (reset! uploads [])
          (let [overlap-capacity (page-pool/reserve-capacity! pool :request-b 7)
                overlapped
                (manager/restore-paged-prefix-overlapped!
                 cache pool :request-b "fixture-paged-v1"
                 [1 2 3 4 5 6 7]
                 {:capacity-reservation overlap-capacity :policy {}})]
            (is (= 4 (:cached-token-count overlapped)))
            (is (= 4 (count @uploads)))
            (is (= 4 (:restored-chunks (manager/stats cache))))
            (is (page-pool/release-capacity! pool overlap-capacity))
            (is (page-pool/release-route! pool :request-b)))
          (let [cancelled? (atom false)
                cancelled-capacity
                (page-pool/reserve-capacity! pool :request-c 7)]
            (with-redefs [gpu/event-complete?
                          (fn [& _]
                            (reset! cancelled? true)
                            true)]
              (is (thrown? CancellationException
                           (manager/restore-paged-prefix-overlapped!
                            cache pool :request-c "fixture-paged-v1"
                            [1 2 3 4 5 6 7]
                            {:capacity-reservation cancelled-capacity
                             :policy {}
                             :cancelled? #(deref cancelled?)}))))
            (is (nil? (page-pool/route pool :request-c)))
            (is (page-pool/release-capacity! pool cancelled-capacity)))))
      (finally
        (.close cache)
        (d/delete-database config)
        (delete-directory! directory)))))

(deftest async-checkpoint-is-bounded-and-does-not-block-the-caller
  (let [directory (Files/createTempDirectory "pretrained-kv-async-manager-"
                                               (make-array java.nio.file.attribute.FileAttribute 0))
        config {:store {:backend :memory :id (random-uuid)}
                :schema-flexibility :write :keep-history? false :value-caps :default}
        model {:n-layers 1 :n-kv 1 :head-dim 2}
        metadata {:continuation/version 1
                  :continuation/model-fingerprint "fixture-async-v1"
                  :continuation/layout (continuation/model-layout model)
                  :continuation/processed-count 2
                  :continuation/pending-token 3
                  :continuation/tokens [1 2 3]}
        state {:continuation/backend :gpu}
        entered (CountDownLatch. 1)
        release-capture (CountDownLatch. 1)
        cache (manager/open-manager config directory {:max-pending-captures 1})]
    (try
      (with-redefs [continuation-gpu/export-gpu-metadata (fn [_] metadata)
                    continuation-gpu/export-gpu-into!
                    (fn [_ destinations]
                      (.countDown entered)
                      (.await release-capture 5 TimeUnit/SECONDS)
                      destinations)]
        (let [first-ticket (manager/checkpoint-gpu-async! cache state)]
          (is (:accepted? first-ticket))
          (is (.await entered 5 TimeUnit/SECONDS))
          (is (not (.isDone ^java.util.concurrent.CompletableFuture
                            (:captured first-ticket)))
              "the caller returned while capture remained blocked")
          (let [queued-ticket (manager/checkpoint-gpu-async! cache state)
                rejected-ticket (manager/checkpoint-gpu-async! cache state)]
            (is (:accepted? queued-ticket))
            (is (false? (:accepted? rejected-ticket))
                "a saturated cache drops work rather than applying backpressure")
            (is (.isCompletedExceptionally ^java.util.concurrent.CompletableFuture
                                           (:captured rejected-ticket)))
            (.countDown release-capture)
            (.get ^java.util.concurrent.CompletableFuture (:captured first-ticket)
                  5 TimeUnit/SECONDS)
            (.get ^java.util.concurrent.CompletableFuture (:published first-ticket)
                  5 TimeUnit/SECONDS)
            (.get ^java.util.concurrent.CompletableFuture (:published queued-ticket)
                  5 TimeUnit/SECONDS)
            (is (some? (manager/lookup cache "fixture-async-v1" [1 2 3])))
            (is (= [2 1]
                   ((juxt :capture-accepted :capture-rejected)
                    (manager/stats cache)))))))
      (finally
        (.countDown release-capture)
        (.close cache)
        (d/delete-database config)
        (delete-directory! directory)))))

(deftest async-chunk-checkpoint-publishes-one-chain-transaction
  (let [directory (Files/createTempDirectory "pretrained-kv-async-chunks-"
                                               (make-array java.nio.file.attribute.FileAttribute 0))
        config {:store {:backend :memory :id (random-uuid)}
                :schema-flexibility :write :keep-history? false :value-caps :default}
        model {:n-layers 1 :n-kv 1 :head-dim 2}
        state {:continuation/backend :gpu
               :continuation/dstate {:model model :maxpos 8 :sess ::session}
               :continuation/model-fingerprint "fixture-async-chunks-v1"
               :continuation/layout (continuation/model-layout model)
               :continuation/max-position 8
               :continuation/processed-count 4
               :continuation/pending-token 5
               :continuation/tokens [1 2 3 4 5]}
        cache (manager/open-manager config directory {:chunk-size 2})]
    (try
      (with-redefs [continuation-gpu/export-gpu-chunk
                    (fn [_ descriptor]
                      (merge descriptor
                             {:chunk/version 1
                              :chunk/model-fingerprint "fixture-async-chunks-v1"
                              :chunk/layout (continuation/model-layout model)
                              :chunk/elements-per-slab 4
                              :chunk/payload (float-array 8)}))]
        (let [ticket (manager/checkpoint-gpu-chunks-async! cache state)
              stored (.get ^java.util.concurrent.CompletableFuture (:captured ticket)
                           5 TimeUnit/SECONDS)
              published (.get ^java.util.concurrent.CompletableFuture (:published ticket)
                              5 TimeUnit/SECONDS)
              lookup (manager/lookup-chunk-prefix
                      cache "fixture-async-chunks-v1" [1 2 3 4 5])]
          (is (:accepted? ticket))
          (is (= 2 (count stored)))
          (is (= 2 (count published)))
          (is (= 4 (:cached-token-count lookup)))))
      (finally
        (.close cache)
        (d/delete-database config)
        (delete-directory! directory)))))

(deftest async-paged-checkpoint-uses-the-same-durable-prefix-chain
  (let [directory (Files/createTempDirectory "pretrained-kv-async-pages-"
                                              (make-array java.nio.file.attribute.FileAttribute 0))
        config {:store {:backend :memory :id (random-uuid)}
                :schema-flexibility :write :keep-history? false :value-caps :default}
        model {:n-layers 1 :n-kv 1 :head-dim 2}
        pool (page-pool/->DevicePagePool
              ::session (attention-state/layout model) 2 4 :half
              {[:key 0] :k0, [:value 0] :v0}
              (atom {:free (sorted-set 2 3)
                     :refcounts {0 1, 1 1}
                     :leases {}
                     :routes {:request {:continuation-id :request
                                        :pages [0 1]
                                        :token-count 4
                                        :start-position 0}}}))
        captured (atom [])
        cache (manager/open-manager config directory {:chunk-size 2})]
    (try
      (with-redefs [page-pool/submit-export-chunk!
                    (fn [_ continuation-id model-fingerprint descriptor]
                      (swap! captured conj [continuation-id descriptor])
                      (merge descriptor
                             {:chunk/version 2
                              :chunk/model-fingerprint model-fingerprint
                              :chunk/layout (continuation/model-layout model)
                              :chunk/elements-per-slab 4
                              :chunk/payload (float-array 8)}))
                    page-pool/chunk-export-complete? (fn [& _] true)
                    page-pool/complete-export-chunk! (fn [_ export] export)]
        (let [ticket (manager/checkpoint-paged-chunks-async!
                      cache pool :request "fixture-paged-v1" [1 2 3 4 5])
              stored (.get ^java.util.concurrent.CompletableFuture (:captured ticket)
                           5 TimeUnit/SECONDS)
              published (.get ^java.util.concurrent.CompletableFuture (:published ticket)
                              5 TimeUnit/SECONDS)
              lookup (manager/lookup-chunk-prefix
                      cache "fixture-paged-v1" [1 2 3 4 5])]
          (is (:accepted? ticket))
          (is (= 2 (count stored) (count published) (count @captured)))
          (is (= 2 (:stored-chunks @(:phase-timings ticket))))
          (is (every? #(not (neg? (double %)))
                      (map @(:phase-timings ticket)
                           [:context-ms :catalog-lookup-ms :device-export-ms
                            :local-persistence-ms :mmap-preparation-ms
                            :capture-total-ms
                            :publication-ms])))
          (is (every? #(not (contains? % :chunk/payload)) stored)
              "completed capture metadata does not retain host staging arrays")
          (is (= [0 2] (mapv (comp :chunk/start second) @captured)))
          (is (= 4 (:cached-token-count lookup)))))
      (finally
        (.close cache)
        (d/delete-database config)
        (delete-directory! directory)))))

(deftest paged-checkpoint-rejects-a-chunk-over-its-host-staging-budget
  (let [directory (Files/createTempDirectory "pretrained-kv-staging-budget-"
                                              (make-array java.nio.file.attribute.FileAttribute 0))
        config {:store {:backend :memory :id (random-uuid)}
                :schema-flexibility :write :keep-history? false :value-caps :default}
        model {:n-layers 1 :n-kv 1 :head-dim 2}
        pool (page-pool/->DevicePagePool
              ::session (attention-state/layout model) 2 4 :half
              {[:key 0] :k0, [:value 0] :v0}
              (atom {:free (sorted-set 2 3)
                     :refcounts {0 1, 1 1}
                     :leases {}
                     :routes {:request {:continuation-id :request
                                        :pages [0 1]
                                        :token-count 4
                                        :start-position 0}}}))
        cache (manager/open-manager
               config directory {:chunk-size 4 :max-chunk-staging-bytes 31})]
    (try
      (let [ticket (manager/checkpoint-paged-chunks-async!
                    cache pool :request "fixture-budget-v1" [1 2 3 4 5])]
        (is (false? (:accepted? ticket)))
        (is (= :staging-byte-budget (:rejection ticket)))
        (is (= 32 (:estimated-staging-bytes ticket)))
        (is (.isCompletedExceptionally (:captured ticket)))
        (is (= {:capture-rejected 1
                :capture-byte-rejected 1
                :capture-queue-depth 0
                :max-chunk-staging-bytes 31}
               (select-keys (manager/stats cache)
                            [:capture-rejected :capture-byte-rejected
                             :capture-queue-depth :max-chunk-staging-bytes]))))
      (finally
        (.close cache)
        (d/delete-database config)
        (delete-directory! directory)))))

(deftest async-paged-checkpoint-polls-before-awaiting-device-download
  (let [directory (Files/createTempDirectory "pretrained-kv-capture-overlap-"
                                              (make-array java.nio.file.attribute.FileAttribute 0))
        config {:store {:backend :memory :id (random-uuid)}
                :schema-flexibility :write :keep-history? false :value-caps :default}
        model {:n-layers 1 :n-kv 1 :head-dim 2}
        pool (page-pool/->DevicePagePool
              ::session (attention-state/layout model) 2 4 :half
              {[:key 0] :k0, [:value 0] :v0}
              (atom {:free (sorted-set 2 3)
                     :refcounts {0 1, 1 1}
                     :leases {}
                     :routes {:request {:continuation-id :request
                                        :pages [0 1]
                                        :token-count 4
                                        :start-position 0}}}))
        submitted (promise)
        complete? (atom false)
        awaits (atom 0)
        cache (manager/open-manager config directory {:chunk-size 4})]
    (try
      (with-redefs [gpu/buffer-view (fn [_ key opts] {:key key :opts opts})
                    gpu/submit-download-ranges-retained!
                    (fn [_ entries resources]
                      (let [event {:entries entries :resources resources}]
                        (deliver submitted event)
                        event))
                    gpu/event-complete? (fn [& _] @complete?)
                    gpu/await-event!
                    (fn [_ event]
                      (swap! awaits inc)
                      (when-not @complete?
                        (throw (ex-info "Download was awaited before a positive poll" {})))
                      (doseq [[_ ^shorts destination
                               {:keys [dst-element elements]}] (:entries event)
                              index (range elements)]
                        (aset destination (+ dst-element index)
                              (Float/floatToFloat16 (float index)))))
                    gpu/event-measurement
                    (fn [& _] {:direction :download :timing-source :device-event
                               :asynchronous? true :bytes 32 :commands 2
                               :elapsed-ns 20 :submit-host-ns 2 :host-wall-ns 25})
                    gpu/release-event!
                    (fn [_ event]
                      (doseq [resource (:resources event)]
                        (.close ^AutoCloseable resource)))]
        (let [ticket (manager/checkpoint-paged-chunks-async!
                      cache pool :request "fixture-overlap-v1" [1 2 3 4 5])]
          (is (:accepted? ticket))
          (is (not= ::timeout (deref submitted 5000 ::timeout)))
          (is (false? (.isDone (:captured ticket))))
          (is (zero? @awaits)
              "capture polling leaves the decoder-owning session available")
          (is (= 1 (:active-leases (page-pool/stats pool))))
          (reset! complete? true)
          (is (= 1 (count (.get (:captured ticket) 5 TimeUnit/SECONDS))))
          (is (= 1 (count (.get (:published ticket) 5 TimeUnit/SECONDS))))
          (is (= 1 @awaits))
          (is (zero? (:active-leases (page-pool/stats pool))))))
      (finally
        (.close cache)
        (d/delete-database config)
        (delete-directory! directory)))))

(deftest write-behind-chunks-publish-only-after-backend-durability
  (let [directory (Files/createTempDirectory "pretrained-kv-write-behind-"
                                             (make-array java.nio.file.attribute.FileAttribute 0))
        config {:store {:backend :memory :id (random-uuid)}
                :schema-flexibility :write :keep-history? false :value-caps :default}
        model {:n-layers 1 :n-kv 1 :head-dim 2}
        state {:continuation/backend :gpu
               :continuation/dstate {:model model :maxpos 8 :sess ::session}
               :continuation/model-fingerprint "fixture-write-behind-v1"
               :continuation/layout (continuation/model-layout model)
               :continuation/max-position 8
               :continuation/processed-count 4
               :continuation/pending-token 5
               :continuation/tokens [1 2 3 4 5]}
        backend-store (new-mem-store (atom {}) {:sync? true})
        entered (CountDownLatch. 1)
        release-write (promise-chan)
        cache (manager/open-manager
               config directory
               {:chunk-size 2
                :chunk-backend-store
                (delayed-assoc-store backend-store entered release-write)})]
    (try
      (with-redefs [continuation-gpu/export-gpu-chunk
                    (fn [_ descriptor]
                      (merge descriptor
                             {:chunk/version 1
                              :chunk/model-fingerprint "fixture-write-behind-v1"
                              :chunk/layout (continuation/model-layout model)
                              :chunk/elements-per-slab 4
                              :chunk/payload (float-array 8)}))]
        (let [ticket (manager/checkpoint-gpu-chunks-async! cache state)
              captured (.get ^java.util.concurrent.CompletableFuture (:captured ticket)
                             5 TimeUnit/SECONDS)]
          (is (= 2 (count captured)) "GPU-to-local capture completes")
          (is (.await entered 5 TimeUnit/SECONDS) "the backend copy started")
          (is (false? (.isDone ^java.util.concurrent.CompletableFuture
                       (:published ticket)))
              "publication waits for authoritative storage")
          (is (zero? (:cached-token-count
                      (manager/lookup-chunk-prefix
                       cache "fixture-write-behind-v1" [1 2 3 4 5])))
              "Datahike cannot expose blobs that exist only on this worker")
          (put! release-write true)
          (let [published (.get ^java.util.concurrent.CompletableFuture
                           (:published ticket) 5 TimeUnit/SECONDS)]
            (is (= 2 (count published)))
            (is (every? #(k/exists? backend-store (:store-key %) {:sync? true})
                        published))
            (is (= 4 (:cached-token-count
                      (manager/lookup-chunk-prefix
                       cache "fixture-write-behind-v1" [1 2 3 4 5])))))))
      (finally
        (put! release-write true)
        (.close cache)
        (d/delete-database config)
        (delete-directory! directory)))))

(deftest generation-schedules-boundaries-and-final-state
  (let [scheduled (atom [])
        state {:continuation/processed-count 0}]
    (with-redefs [continuation-gpu/step-gpu
                  (fn [current]
                    (let [processed (inc (:continuation/processed-count current))]
                      [(assoc current :continuation/processed-count processed) processed]))
                  manager/checkpoint-gpu-async!
                  (fn [_ current]
                    (swap! scheduled conj (:continuation/processed-count current))
                    {:accepted? true})]
      (let [result (manager/advance-gpu-with-checkpoints!
                    nil state 5 {:checkpoint-every 2 :checkpoint-final? true})]
        (is (= [1 2 3 4 5] (:tokens result)))
        (is (= [2 4 5] @scheduled))
        (is (= 3 (count (:checkpoints result))))))))

(deftest manager-close-drains-accepted-checkpoints
  (let [directory (Files/createTempDirectory "pretrained-kv-close-manager-"
                                               (make-array java.nio.file.attribute.FileAttribute 0))
        config {:store {:backend :memory :id (random-uuid)}
                :schema-flexibility :write :keep-history? false :value-caps :default}
        metadata {:continuation/version 1
                  :continuation/model-fingerprint "fixture-close-v1"
                  :continuation/layout
                  (continuation/model-layout {:n-layers 1 :n-kv 1 :head-dim 2})
                  :continuation/processed-count 1
                  :continuation/pending-token 2
                  :continuation/tokens [1 2]}
        cache (manager/open-manager config directory)]
    (try
      (with-redefs [continuation-gpu/export-gpu-metadata (fn [_] metadata)
                    continuation-gpu/export-gpu-into! (fn [_ destinations] destinations)]
        (let [ticket (manager/checkpoint-gpu-async! cache {:continuation/backend :gpu})]
          (.close cache)
          (is (map? (.get ^java.util.concurrent.CompletableFuture (:published ticket)
                          5 TimeUnit/SECONDS))
              "close waits until the accepted catalog publication completes")))
      (finally
        (.close cache)
        (d/delete-database config)
        (delete-directory! directory)))))
