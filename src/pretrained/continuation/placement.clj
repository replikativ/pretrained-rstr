(ns pretrained.continuation.placement
  "Declarative cluster placement for continuation chunks.

  Datahike records desired replicas and observed replicas. This namespace only
  computes reconciliation work; copying bytes and allocating GPU state remain
  worker-local effects. Simmis/Spindel can react to `listen!` notifications, and
  a Yggdrasil Datahike adapter can snapshot the same catalog history."
  (:require [datahike.api :as d]
            [pretrained.continuation.catalog :as catalog])
  (:import [java.nio ByteBuffer]
           [java.nio.charset StandardCharsets]
           [java.security MessageDigest]
           [java.util Date UUID]))

(def replica-states
  "Allowed observed replica lifecycle states."
  #{:kv.replica/copying :kv.replica/ready :kv.replica/failed})

(def ^:private tier-rank
  {:gpu 0 :ram 1 :ssd 2 :object 3})

(defn- stable-id
  [domain values]
  (let [digest (MessageDigest/getInstance "SHA-256")]
    (.update digest (.getBytes ^String domain StandardCharsets/UTF_8))
    (doseq [value values]
      (let [bytes (.getBytes (pr-str value) StandardCharsets/UTF_8)
            length-buffer (doto (ByteBuffer/allocate Long/BYTES)
                            (.putLong (alength bytes)))]
        (.update digest (.array length-buffer))
        (.update digest bytes)))
    (let [bytes (.digest digest)]
      (aset bytes 6 (unchecked-byte (bit-or 0x50 (bit-and 0x0f (aget bytes 6)))))
      (aset bytes 8 (unchecked-byte (bit-or 0x80 (bit-and 0x3f (aget bytes 8)))))
      (let [buffer (ByteBuffer/wrap bytes)]
        (UUID. (.getLong buffer) (.getLong buffer))))))

(defn replica-id
  "Return the stable identity for one model/prefix/node/tier replica."
  [model-fingerprint prefix-hash node tier]
  (stable-id "pretrained-rstr/kv-replica/v1"
             [model-fingerprint prefix-hash node tier]))

(defn demand-id
  "Return the stable identity for one model/prefix/node/tier demand."
  [model-fingerprint prefix-hash node tier]
  (stable-id "pretrained-rstr/kv-demand/v1"
             [model-fingerprint prefix-hash node tier]))

(defn request!
  "Upsert desired local availability for one immutable chunk.

  `request` requires `:model-fingerprint`, `:prefix-hash`, `:node`, and `:tier`.
  `:priority` defaults to zero; `:owner` and `:expires-at` are optional. Repeating
  the same logical request updates its policy fields. Returns a tx report."
  [connection {:keys [model-fingerprint prefix-hash node tier priority owner expires-at]
               :or {priority 0}}]
  (when-not (and (string? model-fingerprint) (uuid? prefix-hash)
                 (string? node) (keyword? tier))
    (throw (ex-info "Placement demand requires model, prefix, node, and tier"
                    {:model-fingerprint model-fingerprint :prefix-hash prefix-hash
                     :node node :tier tier})))
  (d/transact connection
              [(cond-> {:kv/demand-id (demand-id model-fingerprint prefix-hash node tier)
                        :kv/demand-model-fingerprint model-fingerprint
                        :kv/demand-prefix-hash prefix-hash
                        :kv/demand-node node
                        :kv/demand-tier tier
                        :kv/demand-priority (long priority)}
                 owner (assoc :kv/demand-owner (str owner))
                 expires-at (assoc :kv/demand-expires-at expires-at))]))

(defn cancel-request!
  "Retract one placement demand by its stable identity."
  [connection id]
  (d/transact connection [[:db/retractEntity [:kv/demand-id id]]]))

(defn announce-replica!
  "Upsert observed availability for one chunk replica.

  `replica` requires model/prefix/node/tier and a state in `replica-states`.
  Store key, path, and bytes are worker-local observations and are optional."
  [connection {:keys [model-fingerprint prefix-hash node tier state
                      store-key path bytes updated-at]}]
  (when-not (and (string? model-fingerprint) (uuid? prefix-hash)
                 (string? node) (keyword? tier) (contains? replica-states state))
    (throw (ex-info "Replica announcement is incomplete or has an invalid state"
                    {:model-fingerprint model-fingerprint :prefix-hash prefix-hash
                     :node node :tier tier :state state})))
  (d/transact connection
              [(cond-> {:kv/replica-id (replica-id model-fingerprint prefix-hash node tier)
                        :kv/replica-model-fingerprint model-fingerprint
                        :kv/replica-prefix-hash prefix-hash
                        :kv/replica-node node
                        :kv/replica-tier tier
                        :kv/replica-state state
                        :kv/replica-updated-at (or updated-at (Date.))}
                 store-key (assoc :kv/replica-store-key store-key)
                 path (assoc :kv/replica-path (str path))
                 bytes (assoc :kv/replica-bytes (long bytes)))]))

(defn retract-replica!
  "Retract one observed replica by its stable identity."
  [connection id]
  (d/transact connection [[:db/retractEntity [:kv/replica-id id]]]))

(defn demands
  "Return active placement demands for `node` at `now`, highest priority first."
  ([database node] (demands database node (Date.)))
  ([database node ^Date now]
   (->> (d/q '[:find (pull ?e [*])
               :in $ ?node
               :where [?e :kv/demand-node ?node]]
             database node)
        (map first)
        (filter #(or (nil? (:kv/demand-expires-at %))
                     (.after ^Date (:kv/demand-expires-at %) now)))
        (sort-by (juxt (comp - :kv/demand-priority)
                       (comp str :kv/demand-id)))
        vec)))

(defn replicas
  "Return observed replicas for a model-specific chunk prefix."
  [database model-fingerprint prefix-hash]
  (->> (d/q '[:find (pull ?e [*])
              :in $ ?model ?prefix
              :where
              [?e :kv/replica-model-fingerprint ?model]
              [?e :kv/replica-prefix-hash ?prefix]]
            database model-fingerprint prefix-hash)
       (mapv first)))

(defn reconciliation-plan
  "Return satisfied demands and side-effect-free `:ensure-local` actions.

  Every action includes all known ready source replicas. No source means the
  worker must resolve the immutable chunk through a shared/object-store adapter.
  The planner never evicts: eviction needs an explicit cluster policy and lease
  model, rather than treating absence of current demand as permission to delete."
  ([database node] (reconciliation-plan database node (Date.)))
  ([database node now]
   (reduce
    (fn [plan demand]
      (let [model (:kv/demand-model-fingerprint demand)
            prefix (:kv/demand-prefix-hash demand)
            tier (:kv/demand-tier demand)
            candidates (replicas database model prefix)
            ready (->> candidates
                       (filter #(= :kv.replica/ready (:kv/replica-state %)))
                       (sort-by (juxt #(if (= node (:kv/replica-node %)) 0 1)
                                      #(get tier-rank (:kv/replica-tier %) 100)
                                      :kv/replica-node
                                      (comp str :kv/replica-id)))
                       vec)
            local (some #(when (and (= node (:kv/replica-node %))
                                    (= tier (:kv/replica-tier %))) %)
                        ready)]
        (if local
          (update plan :satisfied conj {:demand demand :replica local})
          (update plan :actions conj
                  {:action :ensure-local
                   :demand demand
                   :source (first ready)
                   :sources ready
                   :chunk (catalog/lookup-chunk database model prefix)}))))
    {:node node :satisfied [] :actions []}
    (demands database node now))))

(def placement-attributes
  "Attributes whose transaction changes can affect a reconciliation plan."
  (into #{}
        (map :db/ident)
        (filter #(or (.startsWith (name (:db/ident %)) "demand-")
                     (.startsWith (name (:db/ident %)) "replica-"))
                catalog/schema)))

(defn listen!
  "Notify `consume!` after placement-relevant Datahike transactions.

  The callback receives `{:node :db-after :tx-data}`. It runs on Datahike's
  listener path and therefore must be non-blocking; a Spindel signal update or a
  bounded-queue offer is appropriate. Returns the listener key."
  ([connection node consume!]
   (listen! connection (random-uuid) node consume!))
  ([connection listener-key node consume!]
   (d/listen connection listener-key
             (fn [tx-report]
               (when (some #(contains? placement-attributes (:a %))
                           (:tx-data tx-report))
                 (consume! {:node node
                            :db-after (:db-after tx-report)
                            :tx-data (:tx-data tx-report)}))))
   listener-key))

(defn unlisten!
  "Remove a placement listener."
  [connection listener-key]
  (d/unlisten connection listener-key))
