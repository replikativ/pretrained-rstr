(ns pretrained.continuation.chunk
  "Immutable token-range chunks for durable KV continuations.

  A chunk hash commits to its parent hash and its own processed token ids. This
  is the LMCache prefix-chain property: identical suffix tokens under different
  causal prefixes cannot alias. The continuation's pending token is deliberately
  outside the chain because no KV row has been computed for it yet."
  (:require [pretrained.attention-state :as attention-state])
  (:import [java.nio ByteBuffer]
           [java.nio.charset StandardCharsets]
           [java.security MessageDigest]
           [java.util UUID]))

(def default-chunk-size 256)

(def ^:private hash-domain
  (.getBytes "pretrained-rstr/kv-prefix-chain/v1" StandardCharsets/UTF_8))

(defn- digest-uuid
  [^bytes digest]
  (let [bytes (aclone digest)]
    (aset bytes 6 (unchecked-byte (bit-or 0x50 (bit-and 0x0f (aget bytes 6)))))
    (aset bytes 8 (unchecked-byte (bit-or 0x80 (bit-and 0x3f (aget bytes 8)))))
    (let [buffer (ByteBuffer/wrap bytes)]
      (UUID. (.getLong buffer) (.getLong buffer)))))

(defn- update-long!
  [^MessageDigest digest value]
  (let [buffer (ByteBuffer/allocate Long/BYTES)]
    (.putLong buffer (long value))
    (.update digest (.array buffer))))

(defn chunk-hash
  "Hash one processed-token chunk under `parent-hash`.

  `nil` is the root parent. Token ids and the token count use fixed-width
  big-endian longs, so concatenations cannot be ambiguous. Returns a UUID
  suitable for indexed Datahike attributes."
  [parent-hash tokens]
  (let [digest (MessageDigest/getInstance "SHA-256")
        parent-buffer (ByteBuffer/allocate 16)]
    (.update digest hash-domain)
    (if parent-hash
      (do (.putLong parent-buffer (.getMostSignificantBits ^UUID parent-hash))
          (.putLong parent-buffer (.getLeastSignificantBits ^UUID parent-hash)))
      (do (.putLong parent-buffer 0) (.putLong parent-buffer 0)))
    (.update digest (.array parent-buffer))
    (update-long! digest (count tokens))
    (doseq [token tokens]
      (update-long! digest token))
    (digest-uuid (.digest digest))))

(defn plan
  "Describe the hash chain for `[0, processed-count)`.

  `tokens` must include the processed prefix and may contain the continuation's
  pending token after it. Each result contains `:chunk/index`, `:chunk/start`,
  `:chunk/token-count`, `:chunk/tokens`, `:chunk/parent-hash`, and
  `:chunk/prefix-hash`. The final chunk may be shorter than `chunk-size`."
  ([tokens processed-count] (plan tokens processed-count default-chunk-size))
  ([tokens processed-count chunk-size]
   (let [tokens (vec tokens)
         processed-count (long processed-count)
         chunk-size (long chunk-size)]
     (when-not (pos? chunk-size)
       (throw (ex-info "KV chunk size must be positive" {:chunk-size chunk-size})))
     (when (or (neg? processed-count) (> processed-count (count tokens)))
       (throw (ex-info "Processed token count is outside the token history"
                       {:processed-count processed-count :token-count (count tokens)})))
     (loop [start 0 index 0 parent nil chunks []]
       (if (= start processed-count)
         chunks
         (let [end (min processed-count (+ start chunk-size))
               chunk-tokens (subvec tokens start end)
               prefix-hash (chunk-hash parent chunk-tokens)]
           (recur end (inc index) prefix-hash
                  (conj chunks {:chunk/index index
                                :chunk/start start
                                :chunk/token-count (- end start)
                                :chunk/tokens chunk-tokens
                                :chunk/parent-hash parent
                                :chunk/prefix-hash prefix-hash}))))))))

(defn continuation-plan
  "Plan chunks for a continuation while preserving its pending-token boundary."
  ([state] (continuation-plan state default-chunk-size))
  ([state chunk-size]
   (let [processed (long (:continuation/processed-count state))
         tokens (:continuation/tokens state)
         chunks (plan tokens processed chunk-size)]
     {:chunks chunks
      :processed-count processed
      :pending-token (:continuation/pending-token state)
      :logical-token-count (count tokens)
      :tail-hash (:chunk/prefix-hash (peek chunks))})))

(defn cpu-tensor-chunk
  "Copy one token-range descriptor out of a CPU continuation.

  The returned value is a standalone immutable Boring-friendly chunk with one
  contiguous float payload in declared slab/layer order. A loader can mmap it
  once and slice it into transfers. Only the described token range is copied."
  [state descriptor]
  (when-not (= :cpu (:continuation/backend state))
    (throw (ex-info "cpu-tensor-chunk requires a CPU continuation"
                    {:backend (:continuation/backend state)})))
  (let [layout (get-in state [:continuation/layout :attention-state])
        tensor-groups (into {} (map (fn [[slab tensors]] [(:name slab) [slab tensors]])
                                    (attention-state/tensor-groups state)))
        slabs (attention-state/payload-plan layout (:chunk/token-count descriptor))
        payload-elements (reduce + 0 (map :elements slabs))
        payload (float-array payload-elements)]
    (doseq [{:keys [slab layer element-offset elements]} slabs
            :let [[slab-layout tensors] (get tensor-groups slab)
                  source-offset (* (long (:chunk/start descriptor))
                                   (:elements-per-token slab-layout))
                  source ^floats (nth tensors layer)]]
      (System/arraycopy source (int source-offset) payload
                        (int element-offset) (int elements)))
    (cond-> (merge descriptor
                   {:chunk/version 2
                    :chunk/model-fingerprint (:continuation/model-fingerprint state)
                    :chunk/layout (:continuation/layout state)
                    :chunk/slabs slabs
                    :chunk/payload payload})
      (apply = (map :elements slabs))
      (assoc :chunk/elements-per-slab (:elements (first slabs))))))
