(ns pretrained.continuation
  "Explicit, resumable KV-cache continuations for decoder language models.

  A continuation has one deliberately precise boundary: its KV cache contains
  positions `[0, processed-count)`, while `pending-token` is the token to process
  at `processed-count`.  One step writes that token's K/V row and produces the
  next pending token.  This is the same boundary used by resident GPU decode, so
  CPU and GPU snapshots do not need to persist logits or hidden states."
  (:require [pretrained.attention-state :as attention-state]
            [pretrained.decoder :as dec]
            [pretrained.sampling :as sampling])
  (:import [java.nio ByteOrder]))

(defn model-layout
  "Return the durable attention-state layout descriptor for `model`.

  Conventional KV dimensions remain at the top level for version-1 snapshot
  compatibility. Non-KV architectures can omit them and describe their state
  entirely through `:desc/:attention-state`."
  [model]
  (let [attention-layout (attention-state/layout model)]
    (cond-> {:dtype :float32
             :byte-order (if (= ByteOrder/LITTLE_ENDIAN (ByteOrder/nativeOrder))
                           :little-endian :big-endian)
             :attention-state attention-layout}
      (:n-layers model) (assoc :n-layers (long (:n-layers model)))
      (:n-kv model) (assoc :n-kv (long (:n-kv model)))
      (:head-dim model) (assoc :head-dim (long (:head-dim model)))
      (= :kv (:kind attention-layout)) (assoc :order :layer-kv-token-head-d))))

(defn kv-row-elements
  "Return the number of scalar K or V elements stored per token and layer."
  ^long [model]
  (* (long (:n-kv model)) (long (:head-dim model))))

(defn- require-room!
  [continuation]
  (when-not (< (long (:continuation/processed-count continuation))
               (long (:continuation/max-position continuation)))
    (throw (ex-info "KV continuation has reached its maximum position"
                    {:processed-count (:continuation/processed-count continuation)
                     :max-position (:continuation/max-position continuation)}))))

(defn start-cpu
  "Create a CPU continuation for a non-empty token prompt.

  The prompt is evaluated through its penultimate token. Its final token remains
  pending, making the returned state immediately ready for `step-cpu`. `opts`
  must contain `:max-position`; `:model-fingerprint` is optional until the state
  is exported, but durable callers should supply an exact checkpoint identity.

  Returns a continuation map whose K/V arrays are mutable implementation storage.
  Throws when the prompt is empty or does not fit within `:max-position`."
  [model prompt-ids {:keys [max-position model-fingerprint]}]
  (let [tokens (mapv long prompt-ids)
        prompt-count (count tokens)
        max-position (long max-position)
        _ (when (zero? prompt-count)
            (throw (ex-info "A continuation requires a non-empty prompt" {})))
        _ (when (> prompt-count max-position)
            (throw (ex-info "Prompt does not fit in the KV cache"
                            {:prompt-count prompt-count :max-position max-position})))
        n-layers (long (:n-layers model))
        elements (* max-position (kv-row-elements model))
        keys (vec (repeatedly n-layers #(float-array elements)))
        values (vec (repeatedly n-layers #(float-array elements)))
        processed (dec prompt-count)]
    (doseq [position (range processed)]
      (dec/decode-step model (nth tokens position) position keys values))
    {:continuation/backend :cpu
     :continuation/model model
     :continuation/model-fingerprint model-fingerprint
     :continuation/layout (model-layout model)
     :continuation/max-position max-position
     :continuation/processed-count processed
     :continuation/pending-token (peek tokens)
     :continuation/tokens tokens
     :continuation/keys keys
     :continuation/values values}))

(defn step-cpu
  "Advance a CPU continuation by one token.

  `sampler` receives `[logits vocab]`, as in `pretrained.sampling`. Returns
  `[next-continuation generated-token]`. The K/V arrays are updated in place;
  the continuation map itself is returned as a new value."
  ([continuation] (step-cpu continuation sampling/greedy))
  ([continuation sampler]
   (when-not (= :cpu (:continuation/backend continuation))
     (throw (ex-info "step-cpu requires a CPU continuation"
                     {:backend (:continuation/backend continuation)})))
   (require-room! continuation)
   (let [model (:continuation/model continuation)
         position (long (:continuation/processed-count continuation))
         pending (long (:continuation/pending-token continuation))
         hidden (dec/decode-step model pending position
                                 (:continuation/keys continuation)
                                 (:continuation/values continuation))
         token (long (sampler (dec/lm-logits model hidden) (:vocab model)))]
     [(-> continuation
          (assoc :continuation/processed-count (inc position)
                 :continuation/pending-token token)
          (update :continuation/tokens conj token))
      token])))

(defn advance-cpu
  "Advance a CPU continuation `n` steps.

  Returns `{:continuation state :tokens generated-token-vector}`. The default
  sampler is greedy. Throws before a step that would exceed the cache capacity."
  ([continuation n] (advance-cpu continuation n sampling/greedy))
  ([continuation n sampler]
   (loop [state continuation remaining (long n) tokens []]
     (if (zero? remaining)
       {:continuation state :tokens tokens}
       (let [[next-state token] (step-cpu state sampler)]
         (recur next-state (dec remaining) (conj tokens token)))))))

(defn export-cpu
  "Copy the occupied prefix of a CPU continuation into a portable snapshot.

  Only `processed-count` rows are copied; unused `max-position` capacity is not
  exported. The pending token and logical token sequence are included because
  KV tensors alone are not a complete continuation boundary."
  [continuation]
  (when-not (= :cpu (:continuation/backend continuation))
    (throw (ex-info "export-cpu requires a CPU continuation"
                    {:backend (:continuation/backend continuation)})))
  (let [processed (long (:continuation/processed-count continuation))
        row-size (kv-row-elements (:continuation/model continuation))
        elements (int (* processed row-size))
        copy-prefix (fn [^floats values]
                      (java.util.Arrays/copyOf values elements))]
    {:continuation/version 1
     :continuation/model-fingerprint (:continuation/model-fingerprint continuation)
     :continuation/layout (:continuation/layout continuation)
     :continuation/processed-count processed
     :continuation/pending-token (:continuation/pending-token continuation)
     :continuation/tokens (:continuation/tokens continuation)
     :continuation/keys (mapv copy-prefix (:continuation/keys continuation))
     :continuation/values (mapv copy-prefix (:continuation/values continuation))}))

(defn restore-cpu
  "Restore a CPU continuation from an `export-cpu` snapshot.

  `opts` must contain `:max-position` and may contain `:model-fingerprint`.
  When both the snapshot and caller provide fingerprints they must match.
  Throws for incompatible model layout, capacity or malformed tensor lengths."
  [model snapshot {:keys [max-position model-fingerprint]}]
  (let [expected-layout (model-layout model)
        snapshot-fingerprint (:continuation/model-fingerprint snapshot)
        processed (long (:continuation/processed-count snapshot))
        max-position (long max-position)
        row-size (kv-row-elements model)
        occupied-elements (* processed row-size)
        capacity-elements (* max-position row-size)
        n-layers (long (:n-layers model))
        source-keys (:continuation/keys snapshot)
        source-values (:continuation/values snapshot)]
    (when-not (= 1 (:continuation/version snapshot))
      (throw (ex-info "Unsupported continuation snapshot version"
                      {:version (:continuation/version snapshot)})))
    (when-not (= expected-layout (:continuation/layout snapshot))
      (throw (ex-info "Continuation layout does not match the model"
                      {:expected expected-layout
                       :actual (:continuation/layout snapshot)})))
    (when (and snapshot-fingerprint model-fingerprint
               (not= snapshot-fingerprint model-fingerprint))
      (throw (ex-info "Continuation model fingerprint does not match"
                      {:expected snapshot-fingerprint :actual model-fingerprint})))
    (when (> processed max-position)
      (throw (ex-info "Continuation does not fit in the requested KV capacity"
                      {:processed-count processed :max-position max-position})))
    (when-not (and (= n-layers (count source-keys))
                   (= n-layers (count source-values))
                   (every? #(= occupied-elements (alength ^floats %))
                           (concat source-keys source-values)))
      (throw (ex-info "Continuation tensors do not match their declared layout"
                      {:n-layers n-layers :occupied-elements occupied-elements})))
    (let [restore (fn [^floats source]
                    (let [target (float-array capacity-elements)]
                      (System/arraycopy source 0 target 0 occupied-elements)
                      target))]
      {:continuation/backend :cpu
       :continuation/model model
       :continuation/model-fingerprint (or model-fingerprint snapshot-fingerprint)
       :continuation/layout expected-layout
       :continuation/max-position max-position
       :continuation/processed-count processed
       :continuation/pending-token (:continuation/pending-token snapshot)
       :continuation/tokens (vec (:continuation/tokens snapshot))
       :continuation/keys (mapv restore source-keys)
       :continuation/values (mapv restore source-values)})))
