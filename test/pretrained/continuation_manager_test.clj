(ns pretrained.continuation-manager-test
  (:require [clojure.test :refer [deftest is]]
            [datahike.api :as d]
            [pretrained.continuation :as continuation]
            [pretrained.continuation.gpu :as continuation-gpu]
            [pretrained.continuation.manager :as manager]
            [pretrained.decoder-gpu :as decoder-gpu]
            [raster.gpu.core :as gpu])
  (:import [java.lang.foreign MemorySegment]
           [java.nio.file Files]
           [java.util.concurrent CountDownLatch TimeUnit]))

(defn- delete-directory!
  [directory]
  (doseq [file (reverse (file-seq (.toFile directory)))]
    (Files/deleteIfExists (.toPath file))))

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
