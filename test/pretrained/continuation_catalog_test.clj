(ns pretrained.continuation-catalog-test
  (:require [clojure.test :refer [deftest is]]
            [datahike.api :as d]
            [pretrained.continuation.catalog :as catalog]
            [pretrained.continuation.chunk :as chunk]))

(defn- memory-config
  []
  {:store {:backend :memory :id (random-uuid)}
   :schema-flexibility :write
   :keep-history? false
   :value-caps :default})

(deftest exact-prefix-catalog-roundtrip
  (let [config (memory-config)
        connection (catalog/ensure-database! config)
        prefix (catalog/token-prefix-hash [1 2 3])
        id (random-uuid)
        blob (random-uuid)]
    (try
      (catalog/put! connection
                    {:kv/id id
                     :kv/model-fingerprint "fixture-v1"
                     :kv/prefix-hash prefix
                     :kv/processed-count 2
                     :kv/pending-token 3
                     :kv/logical-token-count 3}
                    {:content-id blob :path "/tmp/fixture.rstrkv" :bytes 4096})
      (let [found (catalog/lookup @connection "fixture-v1" prefix)]
        (is (= id (:kv/id found)))
        (is (= blob (:kv/blob found)))
        (is (= 2 (:kv/processed-count found))))
      (catalog/retract! connection id)
      (is (nil? (catalog/lookup @connection "fixture-v1" prefix)))
      (finally
        (d/release connection)
        (d/delete-database config)))))

(deftest batch-chunk-lookup-and-contiguous-prefix
  (let [config (memory-config)
        connection (catalog/ensure-database! config)
        descriptors (chunk/plan [1 2 3 4 5 6] 6 2)
        stored (mapv (fn [descriptor]
                       (assoc descriptor :store-key (random-uuid) :bytes 128))
                     [(first descriptors) (last descriptors)])]
    (try
      (catalog/put-chunks! connection "fixture-v1" stored)
      (let [replacement (random-uuid)]
        (catalog/put-chunks! connection "fixture-v1"
                             [(assoc (first descriptors)
                                     :store-key replacement :bytes 256)])
        (is (= replacement
               (:kv/store-key
                (first (catalog/lookup-chunks
                        @connection "fixture-v1"
                        [(:chunk/prefix-hash (first descriptors))]))))
            "model plus chain hash is one idempotent logical catalog node"))
      (let [found (catalog/lookup-chunks
                   @connection "fixture-v1" (mapv :chunk/prefix-hash descriptors))]
        (is (= [0 2] (mapv :kv/chunk-index found))
            "one batch query returns present chunks in requested order")
        (is (= [0] (mapv :kv/chunk-index
                         (catalog/longest-prefix descriptors found)))
            "a missing middle chunk prevents a later orphan from loading"))
      (catalog/put-chunks! connection "fixture-v1"
                           [(assoc (second descriptors)
                                   :store-key (random-uuid) :bytes 128)])
      (let [found (catalog/lookup-chunks
                   @connection "fixture-v1" (mapv :chunk/prefix-hash descriptors))]
        (is (= [0 1 2] (mapv :kv/chunk-index
                             (catalog/longest-prefix descriptors found)))))
      (finally
        (d/release connection)
        (d/delete-database config)))))

(deftest ensure-database-upgrades-the-whole-snapshot-schema
  (let [config (memory-config)]
    (try
      (d/create-database (assoc config :initial-tx (take 10 catalog/schema)))
      (let [connection (catalog/ensure-database! config)]
        (try
          (is (= :kv/store-key (:db/ident (d/entity @connection :kv/store-key))))
          (is (= :kv/kind (:db/ident (d/entity @connection :kv/kind))))
          (finally
            (d/release connection))))
      (finally
        (d/delete-database config)))))
