(ns pretrained.continuation.controller.candidates
  "Build request-specific routing candidates from facts and worker observations."
  (:require [pretrained.continuation.catalog :as catalog]
            [pretrained.continuation.chunk :as chunk]
            [pretrained.continuation.controller.protocol :as protocol]
            [pretrained.continuation.controller.router :as router]
            [pretrained.continuation.placement :as placement]))

(def ^:private lower-tier-rank {:ram 0 :ssd 1 :object 2})

(defn- non-negative-number?
  [value]
  (and (number? value) (not (neg? (double value)))))

(defn- positive-number?
  [value]
  (and (number? value) (pos? (double value))))

(defn worker-observation
  "Validate and normalize one ephemeral worker scheduling observation.

  Required fields describe identity, epoch, loaded model fingerprints, queue and
  page capacity, context capacity, measured prefill/first-token costs, device
  upload throughput, and per-tier byte throughput. `:worker/tier-fixed-ms`,
  `:worker/gpu-prefixes`, and `:worker/object-store?` are optional. GPU prefixes
  are keyed by `[model-fingerprint prefix-hash]` and name their resident
  continuation, token count, and bytes."
  [{:worker/keys [id node epoch models online? queue-ms page-size free-pages
                  evictable-pages max-context prefill-ms-per-token first-token-ms
                  gpu-restore-bytes-per-ms tier-throughput-bytes-per-ms
                  tier-fixed-ms gpu-prefixes object-store? sequence]
    :as observation}]
  (when (or (nil? id) (not (string? node)))
    (throw (ex-info "Worker observation requires identity and node"
                    {:observation observation})))
  (when-not (set? models)
    (throw (ex-info "Worker observation models must be a set"
                    {:worker/id id :models models})))
  (doseq [[field value]
          [[:worker/epoch epoch]
           [:worker/sequence (or sequence 0)]
           [:worker/queue-ms queue-ms]
           [:worker/page-size page-size]
           [:worker/free-pages free-pages]
           [:worker/evictable-pages evictable-pages]
           [:worker/max-context max-context]
           [:worker/prefill-ms-per-token prefill-ms-per-token]
           [:worker/first-token-ms first-token-ms]]]
    (when-not (non-negative-number? value)
      (throw (ex-info "Worker observation fields must be non-negative numbers"
                      {:worker/id id :field field :value value}))))
  (doseq [[field value]
          [[:worker/page-size page-size]
           [:worker/max-context max-context]]]
    (when-not (and (integer? value) (pos? value))
      (throw (ex-info "Worker observation extent must be a positive integer"
                      {:worker/id id :field field :value value}))))
  (when-not (and (integer? epoch) (integer? (or sequence 0))
                 (not (neg? epoch))
                 (integer? free-pages) (integer? evictable-pages))
    (throw (ex-info "Worker epoch and page counts must be integers"
                    {:worker/id id})))
  (when-not (positive-number? gpu-restore-bytes-per-ms)
    (throw (ex-info "Worker GPU restore throughput must be positive"
                    {:worker/id id
                     :gpu-restore-bytes-per-ms gpu-restore-bytes-per-ms})))
  (doseq [[tier throughput] tier-throughput-bytes-per-ms]
    (when-not (and (contains? lower-tier-rank tier)
                   (positive-number? throughput))
      (throw (ex-info "Worker tier throughput is invalid"
                      {:worker/id id :tier tier :throughput throughput}))))
  (doseq [[tier fixed-ms] tier-fixed-ms]
    (when-not (and (contains? lower-tier-rank tier)
                   (non-negative-number? fixed-ms))
      (throw (ex-info "Worker tier fixed cost is invalid"
                      {:worker/id id :tier tier :fixed-ms fixed-ms}))))
  (assoc observation
         :worker/epoch (long epoch)
         :worker/sequence (long (or sequence 0))
         :worker/online? (not (false? online?))
         :worker/page-size (long page-size)
         :worker/free-pages (long free-pages)
         :worker/evictable-pages (long evictable-pages)
         :worker/max-context (long max-context)
         :worker/tier-fixed-ms (or tier-fixed-ms {})
         :worker/gpu-prefixes (or gpu-prefixes {})
         :worker/object-store? (true? object-store?)))

(defn- base-candidate
  [request observation]
  {:candidate/worker-id (:worker/id observation)
   :candidate/worker-epoch (:worker/epoch observation)
   :candidate/queue-ms (:worker/queue-ms observation)
   :candidate/prefill-ms-per-token (:worker/prefill-ms-per-token observation)
   :candidate/first-token-ms (:worker/first-token-ms observation)
   :candidate/page-size (:worker/page-size observation)
   :candidate/free-pages (:worker/free-pages observation)
   :candidate/evictable-pages (:worker/evictable-pages observation)
   :candidate/max-context (:worker/max-context observation)
   :candidate/online? (:worker/online? observation)
   :candidate/model-loaded?
   (contains? (:worker/models observation)
              (:request/model-fingerprint request))})

(defn- tier-load-ms
  [observation tier bytes]
  (when-let [throughput (get-in observation
                                [:worker/tier-throughput-bytes-per-ms tier])]
    (+ (double (get-in observation [:worker/tier-fixed-ms tier] 0.0))
       (/ (double bytes) (double throughput)))))

(defn- local-ready-tiers
  [replicas node]
  (into #{}
        (comp (filter #(and (= node (:kv/replica-node %))
                            (= :kv.replica/ready (:kv/replica-state %))))
              (map :kv/replica-tier)
              (filter #(contains? #{:ram :ssd} %)))
        replicas))

(defn- fastest-source
  [observation replicas bytes]
  (let [local (local-ready-tiers replicas (:worker/node observation))
        tiers (cond-> local (:worker/object-store? observation) (conj :object))]
    (->> tiers
         (keep (fn [tier]
                 (when-let [cost (tier-load-ms observation tier bytes)]
                   {:tier tier :load-ms cost})))
         (sort-by (juxt :load-ms (comp lower-tier-rank :tier)))
         first)))

(defn- durable-candidate
  [request observation sources cached-tokens cached-bytes load-ms]
  (let [tier (:tier (apply max-key (comp lower-tier-rank :tier) sources))]
    (merge (base-candidate request observation)
           {:candidate/cache-tier tier
            :candidate/cached-token-count cached-tokens
            :candidate/cached-bytes cached-bytes
            :candidate/prefix-load-ms load-ms
            :candidate/gpu-restore-ms
            (/ (double cached-bytes)
               (double (:worker/gpu-restore-bytes-per-ms observation)))})))

(defn- durable-alternatives
  [request observation matched replicas-by-prefix]
  (loop [remaining matched
         sources []
         cached-tokens 0
         cached-bytes 0
         load-ms 0.0
         alternatives []]
    (if-let [entry (first remaining)]
      (if-let [source (fastest-source
                       observation
                       (get replicas-by-prefix (:kv/prefix-hash entry))
                       (:kv/bytes entry))]
        (let [sources (conj sources source)
              cached-tokens (+ cached-tokens (:kv/token-count entry))
              cached-bytes (+ cached-bytes (:kv/bytes entry))
              load-ms (+ load-ms (:load-ms source))]
          (recur (next remaining) sources cached-tokens cached-bytes load-ms
                 (conj alternatives
                       (durable-candidate request observation sources
                                          cached-tokens cached-bytes load-ms))))
        alternatives)
      alternatives)))

(defn- gpu-alternatives
  [request observation descriptors]
  (keep
   (fn [{:chunk/keys [start token-count prefix-hash]}]
     (let [cached-token-count (+ start token-count)
           resident (get-in observation
                            [:worker/gpu-prefixes
                             [(:request/model-fingerprint request) prefix-hash]])]
       (when (and resident
                  (= cached-token-count (:token-count resident))
                  (non-negative-number? (:bytes resident))
                  (some? (:continuation-id resident)))
         (merge (base-candidate request observation)
                {:candidate/cache-tier :gpu
                 :candidate/cached-token-count cached-token-count
                 :candidate/cached-bytes (long (:bytes resident))
                 :candidate/prefix-load-ms 0.0
                 :candidate/gpu-restore-ms 0.0
                 :candidate/source-continuation-id
                 (:continuation-id resident)}))))
   descriptors))

(defn- miss-alternative
  [request observation]
  (merge (base-candidate request observation)
         {:candidate/cache-tier :none
          :candidate/cached-token-count 0
          :candidate/cached-bytes 0
          :candidate/prefix-load-ms 0.0
          :candidate/gpu-restore-ms 0.0}))

(defn candidates
  "Return ranked executable cache alternatives for every observed worker.

  Durable exact-prefix entries and ready RAM/SSD replicas come from `database`.
  An object-store-enabled worker may fetch every published chunk because catalog
  publication follows the backend durability receipt. GPU residency remains an
  ephemeral worker observation and names the route to fork. GPU, every usable
  durable prefix boundary, and recompute alternatives are ranked using the
  ordinary router TTFT estimator. Retaining lower-tier alternatives lets a
  worker decline a stale GPU location and immediately retry from SSD/object.

  Options accept positive `:chunk-size`, defaulting to the chunk module default."
  ([database request observations]
   (candidates database request observations {}))
  ([database request observations {:keys [chunk-size]
                                   :or {chunk-size chunk/default-chunk-size}}]
   (when-not (and (integer? chunk-size) (pos? chunk-size))
     (throw (ex-info "Candidate chunk size must be positive"
                     {:chunk-size chunk-size})))
   (let [request (protocol/generation-request request)
         observations (mapv worker-observation observations)
         descriptors (chunk/plan (:request/tokens request)
                                 (dec (count (:request/tokens request)))
                                 chunk-size)
         prefixes (mapv :chunk/prefix-hash descriptors)
         entries (catalog/lookup-chunks
                  database (:request/model-fingerprint request) prefixes)
         matched (catalog/longest-prefix descriptors entries)
         replicas (placement/replicas-for-prefixes
                   database (:request/model-fingerprint request) prefixes)]
     (router/rank-candidates
      request
      (mapcat
       (fn [observation]
         (into [(miss-alternative request observation)]
               (concat
                (gpu-alternatives request observation descriptors)
                (durable-alternatives request observation matched replicas))))
       observations)))))
