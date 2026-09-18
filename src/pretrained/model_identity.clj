(ns pretrained.model-identity
  "Stable identities for attention-state compatibility boundaries."
  (:require [boring.core :as boring]
            [pretrained.attention-state :as attention-state])
  (:import [java.io BufferedInputStream File FileInputStream]
           [java.security MessageDigest]
           [java.util HexFormat]))

(defn- sha256-stream
  [^File file]
  (let [digest (MessageDigest/getInstance "SHA-256")
        buffer (byte-array (* 1024 1024))]
    (with-open [input (BufferedInputStream. (FileInputStream. file))]
      (loop []
        (let [read-count (.read input buffer)]
          (when-not (= -1 read-count)
            (.update digest buffer 0 read-count)
            (recur)))))
    (.formatHex (HexFormat/of) (.digest digest))))

(defn- canonical
  [value]
  (cond
    (map? value) [:map (mapv (fn [[key item]] [(canonical key) (canonical item)])
                             (sort-by (comp pr-str key) value))]
    (set? value) [:set (mapv canonical (sort-by pr-str value))]
    (sequential? value) (mapv canonical value)
    :else value))

(def determinism-contracts
  "Numerical determinism levels, matching Raster's NumericalStateManifest."
  #{:bitwise :reproducible-order :toleranced :nondeterministic})

(def ^:private structured-variant-keys #{:name :cache :numerical})

(defn- validate-execution-variant!
  "Return `variant` when it is a keyword or a well-formed structured variant.

  A structured variant names the execution family and records the facts that
  make attention state numerically incompatible across builds: the cache
  storage format per attention-state group and the numerical contract."
  [variant]
  (cond
    (keyword? variant) variant
    (map? variant)
    (let [{:keys [name cache numerical]} variant
          unexpected (remove structured-variant-keys (keys variant))]
      (when (seq unexpected)
        (throw (ex-info "Execution variant declares unsupported keys"
                        {:unsupported (vec unexpected)
                         :supported structured-variant-keys})))
      (when-not (keyword? name)
        (throw (ex-info "Structured execution variant requires a keyword :name"
                        {:variant variant})))
      (when-not (and (map? cache) (seq cache)
                     (every? (fn [[group format]]
                               (and (keyword? group) (map? format)
                                    (keyword? (:dtype format))
                                    (keyword? (:quantization format))))
                             cache))
        (throw (ex-info (str "Execution variant :cache maps each attention-state "
                             "group to {:dtype :quantization}")
                        {:cache cache})))
      (when-not (and (map? numerical) (keyword? (:mode numerical))
                     (contains? determinism-contracts (:determinism numerical)))
        (throw (ex-info (str "Execution variant :numerical requires a keyword :mode "
                             "and a known :determinism")
                        {:numerical numerical :determinism determinism-contracts})))
      variant)
    :else
    (throw (ex-info "Execution variant must be a keyword or a structured map"
                    {:variant variant}))))

(defn numerical-contract
  "Return `{:mode :determinism}` declared by a structured execution variant.

  Keyword variants declare no contract and return nil; callers then use the
  conservative default."
  [variant]
  (when (map? (validate-execution-variant! variant))
    (select-keys (:numerical variant) [:mode :determinism])))

(defn- require-cache-groups!
  "Throw unless a structured variant's `:cache` names exactly the layout's groups."
  [variant layout]
  (when (map? variant)
    (let [declared (set (keys (:cache variant)))
          groups (into #{} (map :id) (attention-state/groups layout))]
      (when-not (= declared groups)
        (throw (ex-info "Execution variant :cache must name every attention-state group"
                        {:declared declared :groups groups})))))
  variant)

(defn require-cache-dtype!
  "Throw unless every group of a structured `variant` stores `dtype`.

  `dtype` is the durable storage dtype of the cache the variant runs with, e.g.
  `:float16` for the paged pool. Keyword variants declare nothing to check."
  [variant dtype]
  (when (map? variant)
    (let [mismatched (into {} (remove #(= dtype (:dtype (val %)))) (:cache variant))]
      (when (seq mismatched)
        (throw (ex-info "Execution variant :cache disagrees with the cache storage dtype"
                        {:storage-dtype dtype :mismatched mismatched})))))
  variant)

(defn compatibility-fingerprint
  "Return a stable SHA-256 identity for reusable attention state.

  `model` supplies its config, architecture descriptor, attention-state layout,
  and local `model.safetensors`. `opts` may instead supply a stable `:weights-id`
  (for example an immutable repository revision), avoiding a local file scan.
  `:execution-variant` identifies numerically incompatible execution formats and
  defaults to `:default`; callers using distinct quantization schemes should name
  them explicitly. It is either a keyword, hashed exactly as before, or a
  structured map `{:name :cache :numerical}` recording cache storage per
  attention-state group and the numerical contract; its `:cache` must name
  exactly the model's attention-state groups. The result is suitable for
  `:model-fingerprint`.

  Throws when neither an explicit weights identity nor a readable weights file is
  available. Computing a local identity streams the weights once and may be slow."
  ([model] (compatibility-fingerprint model {}))
  ([model {:keys [weights-id execution-variant]
           :or {execution-variant :default}}]
   (validate-execution-variant! execution-variant)
   (require-cache-groups! execution-variant (attention-state/layout model))
   (let [weights-file (when-let [directory (:dir model)]
                        (File. (str directory) "model.safetensors"))
         weights-identity
         (or weights-id
             (when (and weights-file (.isFile weights-file))
               (str "sha256:" (sha256-stream weights-file)))
             (throw (ex-info "Model fingerprint requires :weights-id or model.safetensors"
                             {:directory (:dir model)})))
         manifest (canonical
                   {:format-version 1
                    :weights weights-identity
                    :execution-variant execution-variant
                    :architecture (or (:arch model) (get-in model [:desc :arch]))
                    :config (:config model)
                    :descriptor (:desc model)
                    :attention-state (attention-state/layout model)})
         digest (MessageDigest/getInstance "SHA-256")]
     (.update digest ^bytes (boring/encode manifest {:profile :archival}))
     (str "sha256:" (.formatHex (HexFormat/of) (.digest digest))))))
