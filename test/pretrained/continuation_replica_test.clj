(ns pretrained.continuation-replica-test
  (:require [clojure.test :refer [deftest is testing]]
            [datahike.api :as d]
            [pretrained.continuation :as continuation]
            [pretrained.continuation.catalog :as catalog]
            [pretrained.continuation.chunk :as chunk]
            [pretrained.continuation.chunk-store :as chunk-store]
            [pretrained.continuation.placement :as placement]
            [pretrained.continuation.replica :as replica])
  (:import [java.nio.file Files]
           [java.util Comparator]))

(defn- memory-config
  []
  {:store {:backend :memory :id (random-uuid)}
   :schema-flexibility :write
   :keep-history? false
   :value-caps :default})

(defn- delete-tree!
  [directory]
  (when (Files/exists directory (make-array java.nio.file.LinkOption 0))
    (with-open [paths (Files/walk directory (make-array java.nio.file.FileVisitOption 0))]
      (doseq [path (iterator-seq (.iterator (.sorted paths (Comparator/reverseOrder))))]
        (Files/deleteIfExists path)))))

(defn- fixture-chunk
  []
  (merge (first (chunk/plan [11 12] 2 2))
         {:chunk/version 2
          :chunk/model-fingerprint "fixture-v1"
          :chunk/layout (continuation/model-layout
                         {:n-layers 1 :n-kv 1 :head-dim 2})
          :chunk/payload (float-array [1 2 3 4 5 6 7 8])}))

(defn- await-ready
  [connection model prefix]
  (let [deadline (+ (System/nanoTime) (* 5 1000 1000 1000))]
    (loop []
      (let [candidate (some #(when (= "worker-b" (:kv/replica-node %)) %)
                            (placement/replicas @connection model prefix))]
        (cond
          (= :kv.replica/ready (:kv/replica-state candidate)) candidate
          (< (System/nanoTime) deadline) (do (Thread/sleep 10) (recur))
          :else candidate)))))

(deftest tx-driven-executor-copies-and-verifies-a-worker-local-chunk
  (let [config (memory-config)
        connection (catalog/ensure-database! config)
        source-directory (Files/createTempDirectory
                          "pretrained-kv-source-"
                          (make-array java.nio.file.attribute.FileAttribute 0))
        target-directory (Files/createTempDirectory
                          "pretrained-kv-target-"
                          (make-array java.nio.file.attribute.FileAttribute 0))
        source-store (chunk-store/open-store source-directory)
        target-store (chunk-store/open-store target-directory)
        source (replica/konserve-repository source-store)
        target (replica/konserve-repository target-store)
        tensor-chunk (fixture-chunk)
        stored (chunk-store/put! source-store tensor-chunk)
        descriptor (merge tensor-chunk stored)
        prefix (:chunk/prefix-hash tensor-chunk)
        model (:chunk/model-fingerprint tensor-chunk)
        executor (atom nil)]
    (try
      (catalog/put-chunks! connection model [descriptor])
      (placement/announce-replica!
       connection (merge {:model-fingerprint model :prefix-hash prefix
                          :node "worker-a" :tier :ssd
                          :state :kv.replica/ready}
                         stored))
      (placement/announce-replica!
       connection (merge {:model-fingerprint model :prefix-hash prefix
                          :node "worker-0-unmounted" :tier :ssd
                          :state :kv.replica/ready}
                         stored))
      (reset! executor
              (replica/open-executor
               connection "worker-b" :ssd target
               (replica/repository-resolver {"worker-a" source})))
      (placement/request! connection
                          {:model-fingerprint model :prefix-hash prefix
                           :node "worker-b" :tier :ssd :priority 10})
      (testing "the listener falls through an unmounted source and copies to ready"
        (let [ready (await-ready connection model prefix)]
          (is (= :kv.replica/ready (:kv/replica-state ready)))
          (is (= (:store-key stored) (:kv/replica-store-key ready)))
          (is (= (:store-key stored) (:kv/replica-blob ready)))
          (is (chunk-store/stored? target-store (:store-key stored)))))
      (testing "the independently stored payload preserves its content identity"
        (is (= (:store-key stored)
               (chunk-store/content-id
                (chunk-store/read-chunk target-store (:store-key stored))))))
      (finally
        (when-let [running @executor]
          (replica/close-executor! running))
        (d/release connection)
        (d/delete-database config)
        (delete-tree! source-directory)
        (delete-tree! target-directory)))))

(deftest corrupt-content-is-not-published-ready
  (let [config (memory-config)
        connection (catalog/ensure-database! config)
        target-directory (Files/createTempDirectory
                          "pretrained-kv-corrupt-target-"
                          (make-array java.nio.file.attribute.FileAttribute 0))
        target-store (chunk-store/open-store target-directory)
        target (replica/konserve-repository target-store)
        tensor-chunk (fixture-chunk)
        expected-key (chunk-store/content-id tensor-chunk)
        descriptor (assoc tensor-chunk :store-key expected-key :bytes 32)
        prefix (:chunk/prefix-hash tensor-chunk)
        model (:chunk/model-fingerprint tensor-chunk)
        corrupt-source
        (reify replica/ChunkRepository
          (chunk-present? [_ _store-key] true)
          (read-replica-chunk [_ _store-key]
            (assoc tensor-chunk :chunk/payload (float-array [9 9 9 9 9 9 9 9])))
          (write-replica-chunk! [_ _chunk]
            (throw (UnsupportedOperationException.))))]
    (try
      (catalog/put-chunks! connection model [descriptor])
      (placement/announce-replica!
       connection {:model-fingerprint model :prefix-hash prefix
                   :node "worker-a" :tier :ssd :state :kv.replica/ready
                   :store-key expected-key :bytes 32})
      (placement/request! connection
                          {:model-fingerprint model :prefix-hash prefix
                           :node "worker-b" :tier :ssd})
      (let [action (first (:actions
                           (placement/reconciliation-plan @connection "worker-b")))
            result (replica/copy-action!
                    connection "worker-b" :ssd target
                    (constantly corrupt-source) action)
            failed (some #(when (= "worker-b" (:kv/replica-node %)) %)
                         (placement/replicas @connection model prefix))]
        (is (= :failed (:status result)))
        (is (= :kv.replica/failed (:kv/replica-state failed)))
        (is (re-find #"content verification" (:kv/replica-error failed)))
        (is (not (chunk-store/stored? target-store expected-key))))
      (finally
        (d/release connection)
        (d/delete-database config)
        (delete-tree! target-directory)))))
