(ns pretrained.continuation.catalog
  "Datahike catalog for durable continuation blobs.

  Queryable identity and policy live in datoms. Tensor bytes remain in external,
  mmap-friendly files named by `:db.type/store-ref`; Datahike is their root set."
  (:require [clojure.edn :as edn]
            [datahike.api :as d])
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
    :db/cardinality :db.cardinality/one}

   ;; The algorithm that content-addresses a node's blobs. It is per node, not
   ;; per fingerprint, so a later algorithm can publish under an existing
   ;; fingerprint; nodes without it predate the attribute and use Hasch v3.
   {:db/ident :kv/content-algorithm :db/valueType :db.type/keyword
    :db/cardinality :db.cardinality/one}

   ;; A node of a layout with several retention groups stores one part per
   ;; group instead of a node-level blob. Parts are components of their node,
   ;; so retracting the node retracts them, and each part's store-ref is a
   ;; garbage-collection root. Such nodes carry no :kv/store-key, so a path that
   ;; does not understand parts fails instead of loading one part as a chunk.
   {:db/ident :kv/parts :db/valueType :db.type/ref
    :db/cardinality :db.cardinality/many :db/isComponent true}
   {:db/ident :kv.part/id :db/valueType :db.type/uuid
    :db/cardinality :db.cardinality/one :db/unique :db.unique/identity}
   {:db/ident :kv.part/group :db/valueType :db.type/keyword
    :db/cardinality :db.cardinality/one}
   {:db/ident :kv.part/store-key :db/valueType :db.type/uuid
    :db/cardinality :db.cardinality/one}
   {:db/ident :kv.part/blob :db/valueType :db.type/store-ref
    :db/cardinality :db.cardinality/one}
   {:db/ident :kv.part/bytes :db/valueType :db.type/long
    :db/cardinality :db.cardinality/one}

   ;; The durable layout of a compatibility fingerprint: the facts a numerical
   ;; state manifest needs without opening chunk blobs. The layout is written
   ;; once, by compare-and-swap from nil in the same transaction as the first
   ;; chunks, so racing publishers with different layouts cannot both succeed
   ;; and a failed swap publishes none of its chunks. Shard and chunk-grid
   ;; facts never enter the layout; they are part coordinates.
   {:db/ident :kv.layout/fingerprint :db/valueType :db.type/string
    :db/cardinality :db.cardinality/one :db/unique :db.unique/identity}
   {:db/ident :kv.layout/edn :db/valueType :db.type/string
    :db/cardinality :db.cardinality/one}
   {:db/ident :kv.layout/numerical-mode :db/valueType :db.type/keyword
    :db/cardinality :db.cardinality/one}
   {:db/ident :kv.layout/determinism :db/valueType :db.type/keyword
    :db/cardinality :db.cardinality/one}])

(def ^:private numerical-modes
  {:float32 :fp32-kv :float16 :fp16-kv})

(defn- digest-uuid
  [^String domain value]
  (let [digest (MessageDigest/getInstance "SHA-256")]
    (.update digest (.getBytes domain java.nio.charset.StandardCharsets/UTF_8))
    (.update digest (.getBytes (pr-str value) java.nio.charset.StandardCharsets/UTF_8))
    (let [bytes (.digest digest)]
      (aset bytes 6 (unchecked-byte (bit-or 0x50 (bit-and 0x0f (aget bytes 6)))))
      (aset bytes 8 (unchecked-byte (bit-or 0x80 (bit-and 0x3f (aget bytes 8)))))
      (let [result (ByteBuffer/wrap bytes)]
        (UUID. (.getLong result) (.getLong result))))))

(defn part-id
  "Return the stable identity of one stored part of a catalog node.

  `coordinates` is a canonical map, currently `{:group g}`; a later shard key
  extends the map without changing how existing parts are named."
  [node-id coordinates]
  (digest-uuid "pretrained-rstr/kv-part/v1" [node-id (into (sorted-map) coordinates)]))

(defn layout-record
  "Return the catalog facts describing `durable-layout` for `model-fingerprint`.

  `durable-layout` is a chunk's `:chunk/layout`. Its canonical EDN is stored so a
  manifest can be projected without opening a blob. `numerical-contract`, from a
  structured execution variant, overrides the dtype-derived mode and the
  conservative `:toleranced` determinism."
  ([model-fingerprint durable-layout]
   (layout-record model-fingerprint durable-layout nil))
  ([model-fingerprint durable-layout numerical-contract]
  (let [mode (or (:mode numerical-contract)
                 (get numerical-modes (:dtype durable-layout)))]
    (when-not mode
      (throw (ex-info "Durable chunk layout has an unsupported storage dtype"
                      {:model-fingerprint model-fingerprint
                       :dtype (:dtype durable-layout)})))
    {:kv.layout/fingerprint model-fingerprint
     :kv.layout/edn (binding [*print-length* nil *print-level* nil
                              *print-namespace-maps* false]
                      (pr-str durable-layout))
     :kv.layout/numerical-mode mode
     :kv.layout/determinism (or (:determinism numerical-contract) :toleranced)})))

(defn lookup-layout
  "Return the recorded durable layout for `model-fingerprint`, or nil.

  The result contains the catalog facts and the parsed `:layout`. A fingerprint
  entity whose layout has not been swapped in yet counts as absent."
  [database model-fingerprint]
  (when-let [record (ffirst
                     (d/q '[:find (pull ?e [*])
                            :in $ ?fingerprint
                            :where [?e :kv.layout/fingerprint ?fingerprint]]
                          database model-fingerprint))]
    (when-let [edn-text (:kv.layout/edn record)]
      (assoc record :layout (edn/read-string edn-text)))))

(def ^:private contract-keys
  [:kv.layout/numerical-mode :kv.layout/determinism])

(defn- layout-tx!
  "Return transaction data recording the layout of `chunks`.

  A compatibility fingerprint names numerically interchangeable state, so every
  chunk published under it shares one durable layout; a mismatch means the
  fingerprint omitted an execution-relevant fact, such as the cache storage
  format. The same holds for the numerical contract, which comes from the
  execution variant the fingerprint hashes. The
  first publisher therefore owns these facts, including the storage dtype: a
  paged FP16 publisher cannot join a fingerprint an FP32 publisher recorded.
  The first publication creates the fingerprint entity and returns a
  compare-and-swap from nil, which fails, together with the chunks it is
  transacted with, if another publisher recorded a layout first."
  [connection model-fingerprint chunks numerical-contract]
  (let [layouts (into #{} (keep :chunk/layout) chunks)]
    (when (> (count layouts) 1)
      (throw (ex-info "Chunks published under one fingerprint disagree on layout"
                      {:model-fingerprint model-fingerprint :layouts layouts})))
    (when-let [durable-layout (first layouts)]
      (let [record (layout-record model-fingerprint durable-layout numerical-contract)
            existing (lookup-layout @connection model-fingerprint)]
        (cond
          (nil? existing)
          (do
            ;; Idempotent: creates the entity the swap below names.
            (d/transact connection [{:kv.layout/fingerprint model-fingerprint}])
            [[:db/cas [:kv.layout/fingerprint model-fingerprint]
              :kv.layout/edn nil (:kv.layout/edn record)]
             (dissoc record :kv.layout/edn)])

          (and (= (:layout existing) durable-layout)
               (= (select-keys existing contract-keys)
                  (select-keys record contract-keys)))
          []

          (= (:layout existing) durable-layout)
          (throw (ex-info "Fingerprint already has a different numerical contract"
                          {:model-fingerprint model-fingerprint
                           :recorded (select-keys existing contract-keys)
                           :published (select-keys record contract-keys)}))

          :else
          (throw (ex-info "Fingerprint already has a different durable layout"
                          {:model-fingerprint model-fingerprint
                           :recorded (:layout existing)
                           :published durable-layout})))))))

(defn- cas-failure?
  [error]
  (some #(= :transact/cas (:error (ex-data %)))
        (take-while some? (iterate ex-cause error))))

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

  `chunks` pairs a chain descriptor with `:store-key` and `:bytes`, or, for a
  layout with several retention groups, with `:parts` of
  `{:group :store-key :bytes}`; all parts of a node commit together. Content ids
  are used as entity identities, making retrying the same publication idempotent.
  The local Konserve object is durable before this function is called. With
  `:content-algorithm`, chunks carrying `:chunk/layout` also record the
  fingerprint's durable layout in the same transaction; `:numerical-contract`
  is the structured execution variant's `{:mode :determinism}`, if any."
  ([connection model-fingerprint chunks]
   (put-chunks! connection model-fingerprint chunks {}))
  ([connection model-fingerprint chunks {:keys [content-algorithm numerical-contract]}]
  (let [created-at (java.util.Date.)
        tx-data
        (mapv (fn [{:chunk/keys [index start token-count parent-hash prefix-hash]
                    :keys [store-key bytes parts]}]
                (let [node-id (chunk-entry-id model-fingerprint prefix-hash)]
                  (cond-> {:kv/id node-id
                           :kv/kind :kv.kind/chunk
                           :kv/model-fingerprint model-fingerprint
                           :kv/prefix-hash prefix-hash
                           :kv/chunk-index (long index)
                           :kv/start-token (long start)
                           :kv/token-count (long token-count)
                           :kv/created-at created-at}
                    parent-hash (assoc :kv/parent-hash parent-hash)
                    content-algorithm (assoc :kv/content-algorithm content-algorithm)
                    (seq parts)
                    (assoc :kv/parts
                           (mapv (fn [{:keys [group store-key bytes]}]
                                   {:kv.part/id (part-id node-id {:group group})
                                    :kv.part/group group
                                    :kv.part/store-key store-key
                                    :kv.part/blob store-key
                                    :kv.part/bytes (long bytes)})
                                 parts))
                    (empty? parts)
                    (assoc :kv/store-key store-key
                           :kv/blob store-key
                           :kv/bytes (long bytes)))))
              chunks)
        layout-data #(when (and content-algorithm (seq tx-data))
                       (layout-tx! connection model-fingerprint chunks
                                   numerical-contract))]
    (when (seq tx-data)
      (try
        (d/transact connection (into (vec (layout-data)) tx-data))
        (catch Exception error
          ;; Another first publisher recorded the layout between our read and
          ;; our swap. Re-reading turns an equal layout into no layout data
          ;; and a different one into the mismatch error.
          (if (cas-failure? error)
            (d/transact connection (into (vec (layout-data)) tx-data))
            (throw error))))))))

(defn lookup-chunks
  "Batch lookup `prefix-hashes`, returning entries in the requested chain order.

  Missing chunks are omitted. The caller can use `longest-prefix` to retain only
  the contiguous chain from its root; this is one Datahike query, not one query
  per token chunk."
  [database model-fingerprint prefix-hashes]
  (let [requested (vec prefix-hashes)
        found (if (seq requested)
                (d/q '[:find ?prefix (pull ?e [* {:kv/parts [*]}])
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
   (d/q '[:find (pull ?e [* {:kv/parts [*]}])
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
