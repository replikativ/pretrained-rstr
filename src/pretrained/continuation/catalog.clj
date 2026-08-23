(ns pretrained.continuation.catalog
  "Datahike catalog for durable continuation blobs.

  Queryable identity and policy live in datoms. Tensor bytes remain in external,
  mmap-friendly files named by `:db.type/store-ref`; Datahike is their root set."
  (:require [datahike.api :as d])
  (:import [java.nio ByteBuffer]
           [java.security MessageDigest]
           [java.util UUID]))

(defn token-prefix-hash
  "Return a content UUID for an ordered token-id sequence."
  [tokens]
  (let [digest (MessageDigest/getInstance "SHA-256")
        buffer (ByteBuffer/allocate 8)]
    (doseq [token tokens]
      (.clear buffer)
      (.putLong buffer (long token))
      (.update digest (.array buffer)))
    (let [bytes (.digest digest)]
      (aset bytes 6 (unchecked-byte (bit-or 0x50 (bit-and 0x0f (aget bytes 6)))))
      (aset bytes 8 (unchecked-byte (bit-or 0x80 (bit-and 0x3f (aget bytes 8)))))
      (let [bb (ByteBuffer/wrap bytes)]
        (UUID. (.getLong bb) (.getLong bb))))))

(defn chunk-entry-id
  "Return the stable catalog identity for a model-specific prefix-chain node."
  [model-fingerprint ^UUID prefix-hash]
  (let [digest (MessageDigest/getInstance "SHA-256")
        model-bytes (.getBytes ^String model-fingerprint
                               java.nio.charset.StandardCharsets/UTF_8)
        buffer (ByteBuffer/allocate 16)]
    (.update digest (.getBytes "pretrained-rstr/kv-catalog-node/v1"
                               java.nio.charset.StandardCharsets/UTF_8))
    (.update digest model-bytes)
    (.putLong buffer (.getMostSignificantBits prefix-hash))
    (.putLong buffer (.getLeastSignificantBits prefix-hash))
    (.update digest (.array buffer))
    (let [bytes (.digest digest)]
      (aset bytes 6 (unchecked-byte (bit-or 0x50 (bit-and 0x0f (aget bytes 6)))))
      (aset bytes 8 (unchecked-byte (bit-or 0x80 (bit-and 0x3f (aget bytes 8)))))
      (let [result (ByteBuffer/wrap bytes)]
        (UUID. (.getLong result) (.getLong result))))))

(def schema
  [{:db/ident :kv/id :db/valueType :db.type/uuid
    :db/cardinality :db.cardinality/one :db/unique :db.unique/identity}
   {:db/ident :kv/model-fingerprint :db/valueType :db.type/string
    :db/cardinality :db.cardinality/one :db/index true}
   {:db/ident :kv/prefix-hash :db/valueType :db.type/uuid
    :db/cardinality :db.cardinality/one :db/index true}
   {:db/ident :kv/processed-count :db/valueType :db.type/long
    :db/cardinality :db.cardinality/one}
   {:db/ident :kv/pending-token :db/valueType :db.type/long
    :db/cardinality :db.cardinality/one}
   {:db/ident :kv/logical-token-count :db/valueType :db.type/long
    :db/cardinality :db.cardinality/one}
   {:db/ident :kv/bytes :db/valueType :db.type/long
    :db/cardinality :db.cardinality/one}
   {:db/ident :kv/blob :db/valueType :db.type/store-ref
    :db/cardinality :db.cardinality/one}
   {:db/ident :kv/path :db/valueType :db.type/string
    :db/cardinality :db.cardinality/one}
   {:db/ident :kv/created-at :db/valueType :db.type/instant
    :db/cardinality :db.cardinality/one}
   {:db/ident :kv/kind :db/valueType :db.type/keyword
    :db/cardinality :db.cardinality/one :db/index true}
   {:db/ident :kv/parent-hash :db/valueType :db.type/uuid
    :db/cardinality :db.cardinality/one :db/index true}
   {:db/ident :kv/chunk-index :db/valueType :db.type/long
    :db/cardinality :db.cardinality/one}
   {:db/ident :kv/start-token :db/valueType :db.type/long
    :db/cardinality :db.cardinality/one}
   {:db/ident :kv/token-count :db/valueType :db.type/long
    :db/cardinality :db.cardinality/one}
   {:db/ident :kv/store-key :db/valueType :db.type/uuid
    :db/cardinality :db.cardinality/one :db/index true}

   ;; Cluster placement is deliberately separate from the immutable chunk entity.
   ;; A store key/path is meaningful only together with the worker and tier that
   ;; can serve it.
   {:db/ident :kv/replica-id :db/valueType :db.type/uuid
    :db/cardinality :db.cardinality/one :db/unique :db.unique/identity}
   {:db/ident :kv/replica-model-fingerprint :db/valueType :db.type/string
    :db/cardinality :db.cardinality/one :db/index true}
   {:db/ident :kv/replica-prefix-hash :db/valueType :db.type/uuid
    :db/cardinality :db.cardinality/one :db/index true}
   {:db/ident :kv/replica-node :db/valueType :db.type/string
    :db/cardinality :db.cardinality/one :db/index true}
   {:db/ident :kv/replica-tier :db/valueType :db.type/keyword
    :db/cardinality :db.cardinality/one :db/index true}
   {:db/ident :kv/replica-state :db/valueType :db.type/keyword
    :db/cardinality :db.cardinality/one :db/index true}
   {:db/ident :kv/replica-store-key :db/valueType :db.type/uuid
    :db/cardinality :db.cardinality/one}
   {:db/ident :kv/replica-blob :db/valueType :db.type/store-ref
    :db/cardinality :db.cardinality/one}
   {:db/ident :kv/replica-path :db/valueType :db.type/string
    :db/cardinality :db.cardinality/one}
   {:db/ident :kv/replica-bytes :db/valueType :db.type/long
    :db/cardinality :db.cardinality/one}
   {:db/ident :kv/replica-updated-at :db/valueType :db.type/instant
    :db/cardinality :db.cardinality/one}
   {:db/ident :kv/replica-error :db/valueType :db.type/string
    :db/cardinality :db.cardinality/one}

   {:db/ident :kv/demand-id :db/valueType :db.type/uuid
    :db/cardinality :db.cardinality/one :db/unique :db.unique/identity}
   {:db/ident :kv/demand-model-fingerprint :db/valueType :db.type/string
    :db/cardinality :db.cardinality/one :db/index true}
   {:db/ident :kv/demand-prefix-hash :db/valueType :db.type/uuid
    :db/cardinality :db.cardinality/one :db/index true}
   {:db/ident :kv/demand-node :db/valueType :db.type/string
    :db/cardinality :db.cardinality/one :db/index true}
   {:db/ident :kv/demand-tier :db/valueType :db.type/keyword
    :db/cardinality :db.cardinality/one :db/index true}
   {:db/ident :kv/demand-priority :db/valueType :db.type/long
    :db/cardinality :db.cardinality/one}
   {:db/ident :kv/demand-owner :db/valueType :db.type/string
    :db/cardinality :db.cardinality/one}
   {:db/ident :kv/demand-expires-at :db/valueType :db.type/instant
    :db/cardinality :db.cardinality/one}])

(defn ensure-database!
  "Create a Datahike catalog at `config` when absent, then return a connection."
  [config]
  (let [existing? (d/database-exists? config)]
    (when-not existing?
      (d/create-database (assoc config :initial-tx schema)))
    (let [connection (d/connect config)]
      ;; Upgrade catalogs created by the whole-snapshot prototype. Schema
      ;; entities are idempotent upserts by :db/ident.
      (when existing?
        (d/transact connection schema))
      connection)))

(defn put!
  "Publish one completely written continuation blob in the catalog.

  `entry` supplies model/prefix identity and snapshot counts. `stored` is the
  result of `continuation.store/write-snapshot!`. Returns the transaction report."
  [connection entry stored]
  (d/transact connection
              [(merge {:kv/id (or (:kv/id entry) (random-uuid))
                       :kv/blob (:content-id stored)
                       :kv/path (:path stored)
                       :kv/bytes (:bytes stored)
                       :kv/created-at (java.util.Date.)}
                      entry)]))

(defn lookup
  "Return a catalog entry for exact `model-fingerprint` and `prefix-hash`."
  [database model-fingerprint prefix-hash]
  (ffirst
   (d/q '[:find (pull ?e [*])
          :in $ ?model ?prefix
          :where
          [?e :kv/model-fingerprint ?model]
          [?e :kv/prefix-hash ?prefix]]
        database model-fingerprint prefix-hash)))

(defn put-chunks!
  "Publish completely written immutable chunks in one Datahike transaction.

  `chunks` pairs a chain descriptor with `:store-key` and `:bytes`. Content ids
  are used as entity identities, making retrying the same publication idempotent.
  The local Konserve object is durable before this function is called."
  [connection model-fingerprint chunks]
  (let [created-at (java.util.Date.)
        tx-data
        (mapv (fn [{:chunk/keys [index start token-count parent-hash prefix-hash]
                    :keys [store-key bytes]}]
                (cond-> {:kv/id (chunk-entry-id model-fingerprint prefix-hash)
                         :kv/kind :kv.kind/chunk
                         :kv/model-fingerprint model-fingerprint
                         :kv/prefix-hash prefix-hash
                         :kv/chunk-index (long index)
                         :kv/start-token (long start)
                         :kv/token-count (long token-count)
                         :kv/store-key store-key
                         :kv/blob store-key
                         :kv/bytes (long bytes)
                         :kv/created-at created-at}
                  parent-hash (assoc :kv/parent-hash parent-hash)))
              chunks)]
    (when (seq tx-data)
      (d/transact connection tx-data))))

(defn lookup-chunks
  "Batch lookup `prefix-hashes`, returning entries in the requested chain order.

  Missing chunks are omitted. The caller can use `longest-prefix` to retain only
  the contiguous chain from its root; this is one Datahike query, not one query
  per token chunk."
  [database model-fingerprint prefix-hashes]
  (let [requested (vec prefix-hashes)
        found (if (seq requested)
                (d/q '[:find ?prefix (pull ?e [*])
                       :in $ ?model [?prefix ...]
                       :where
                       [?e :kv/model-fingerprint ?model]
                       [?e :kv/prefix-hash ?prefix]
                       [?e :kv/kind :kv.kind/chunk]]
                     database model-fingerprint requested)
                [])
        by-hash (into {} found)]
    (vec (keep by-hash requested))))

(defn lookup-chunk
  "Return one immutable chunk entry for exact model and chain-prefix identity."
  [database model-fingerprint prefix-hash]
  (ffirst
   (d/q '[:find (pull ?e [*])
          :in $ ?model ?prefix
          :where
          [?e :kv/model-fingerprint ?model]
          [?e :kv/prefix-hash ?prefix]
          [?e :kv/kind :kv.kind/chunk]]
        database model-fingerprint prefix-hash)))

(defn longest-prefix
  "Return the longest root-contiguous portion of `descriptors` present in `entries`.

  Besides presence, parent hash, start offset, and chunk size must agree. A
  corrupt or stale middle entry therefore cannot cause later chunks to load."
  [descriptors entries]
  (let [by-hash (into {} (map (juxt :kv/prefix-hash identity)) entries)]
    (loop [remaining descriptors parent nil expected-start 0 matched []]
      (if-let [descriptor (first remaining)]
        (let [entry (get by-hash (:chunk/prefix-hash descriptor))]
          (if (and entry
                   (= parent (:chunk/parent-hash descriptor)
                      (:kv/parent-hash entry))
                   (= expected-start (:chunk/start descriptor)
                      (:kv/start-token entry))
                   (= (:chunk/token-count descriptor) (:kv/token-count entry)))
            (recur (next remaining) (:chunk/prefix-hash descriptor)
                   (+ expected-start (:chunk/token-count descriptor))
                   (conj matched entry))
            matched))
        matched))))

(defn retract!
  "Retract a catalog entity, making its external store-ref collectable."
  [connection id]
  (d/transact connection [[:db/retractEntity [:kv/id id]]]))
