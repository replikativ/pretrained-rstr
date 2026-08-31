(ns pretrained.numerical-memory-demo
  "Model-free demonstration of the Datahike/Konserve numerical-memory path."
  (:require [datahike.api :as d]
            [pretrained.continuation :as continuation]
            [pretrained.continuation.chunk-store :as chunk-store]
            [pretrained.continuation.manager :as manager])
  (:import [java.nio.file Files Path]
           [java.nio.file.attribute FileAttribute]))

(def ^:private model
  {:n-layers 2 :n-kv 2 :head-dim 4})

(def ^:private fingerprint
  "numerical-memory-demo/2-layer-gqa-f32-v1")

(defn- values
  [n offset]
  (let [result (float-array n)]
    (dotimes [i n]
      (aset result i (float (+ offset (/ i 1000.0)))))
    result))

(defn- fixture-continuation
  [processed-count]
  (let [row-elements (* (:n-kv model) (:head-dim model))
        tensor-elements (* processed-count row-elements)]
    {:continuation/backend :cpu
     :continuation/model-fingerprint fingerprint
     :continuation/layout (continuation/model-layout model)
     :continuation/processed-count processed-count
     :continuation/pending-token processed-count
     :continuation/tokens (mapv long (range (inc processed-count)))
     :continuation/keys
     (mapv #(values tensor-elements (* 10.0 %)) (range (:n-layers model)))
     :continuation/values
     (mapv #(values tensor-elements (+ 100.0 (* 10.0 %)))
           (range (:n-layers model)))}))

(defn- memory-config
  []
  {:store {:backend :memory :id (random-uuid)}
   :schema-flexibility :write
   :keep-history? false
   :value-caps :default})

(defn run-local!
  "Store, catalog, deduplicate, query, and mmap a synthetic numerical checkpoint.

  `directory` receives the mmap-compatible Konserve chunks and remains owned by
  the caller. The demo uses an ephemeral Datahike catalog and requires no model,
  network access, or GPU."
  [directory]
  (let [config (memory-config)
        cache (manager/open-manager config directory {:chunk-size 256})
        state (fixture-continuation 512)]
    (try
      (let [stored (manager/checkpoint-cpu-chunks! cache state)
            repeated (manager/checkpoint-cpu-chunks! cache state)
            lookup (manager/lookup-chunk-prefix
                    cache fingerprint (:continuation/tokens state))
            first-entry (first (:matched lookup))
            mmap-summary
            (chunk-store/with-mmap-payload
             (manager/local-chunk-store cache)
             (:kv/store-key first-entry)
             #(select-keys % [:element-type :element-count :byte-order]))]
        {:processed-tokens 512
         :chunk-size 256
         :stored-chunks (count stored)
         :deduplicated-on-repeat? (empty? repeated)
         :catalog-matches (count (:matched lookup))
         :cached-tokens (:cached-token-count lookup)
         :stored-bytes (reduce + 0 (map :bytes stored))
         :mmap-payload mmap-summary
         :cache-metrics (manager/stats cache)})
      (finally
        (.close cache)
        (d/delete-database config)))))

(defn- delete-tree!
  [^Path root]
  (when (Files/exists root (make-array java.nio.file.LinkOption 0))
    (with-open [paths (Files/walk root (make-array java.nio.file.FileVisitOption 0))]
      (doseq [path (reverse (iterator-seq (.iterator paths)))]
        (Files/deleteIfExists path)))))

(defn -main
  [& [directory]]
  (if directory
    (let [path (.toPath (java.io.File. directory))]
      (Files/createDirectories path (make-array FileAttribute 0))
      (prn (assoc (run-local! path) :directory (.toString path))))
    (let [path (Files/createTempDirectory
                "pretrained-numerical-memory-" (make-array FileAttribute 0))]
      (try
        (prn (run-local! path))
        (finally
          (delete-tree! path))))))
