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

(defn compatibility-fingerprint
  "Return a stable SHA-256 identity for reusable attention state.

  `model` supplies its config, architecture descriptor, attention-state layout,
  and local `model.safetensors`. `opts` may instead supply a stable `:weights-id`
  (for example an immutable repository revision), avoiding a local file scan.
  `:execution-variant` identifies numerically incompatible execution formats and
  defaults to `:default`; callers using distinct quantization schemes should name
  them explicitly. The result is suitable for `:model-fingerprint`.

  Throws when neither an explicit weights identity nor a readable weights file is
  available. Computing a local identity streams the weights once and may be slow."
  ([model] (compatibility-fingerprint model {}))
  ([model {:keys [weights-id execution-variant]
           :or {execution-variant :default}}]
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
