(ns pretrained.continuation.gpu
  "GPU-resident KV continuation state and prefix transfer.

  The logical boundary is shared with `pretrained.continuation`: K/V contains
  `[0, processed-count)`, and the pending token for `processed-count` is also
  resident in the decoder's `r0` buffer."
  (:require [pretrained.attention-state :as attention-state]
            [pretrained.continuation :as continuation]
            [pretrained.decoder-gpu :as decoder-gpu]
            [raster.gpu.core :as gpu])
  (:import [java.lang.foreign MemorySegment]))

(defn- host-elements
  [source]
  (if (instance? MemorySegment source)
    (quot (.byteSize ^MemorySegment source) 4)
    (alength ^floats source)))

(defn- allocate-tensor-groups
  [layout token-count]
  (reduce (fn [state slab]
            (assoc state (:tensor-key slab)
                   (vec (repeatedly (:count slab)
                                    #(float-array (* (long token-count)
                                                     (:elements-per-token slab)))))))
          {}
          (:slabs layout)))

(defn- require-room!
  [state]
  (when-not (< (long (:continuation/processed-count state))
               (long (:continuation/max-position state)))
    (throw (ex-info "GPU continuation has reached its maximum position"
                    {:processed-count (:continuation/processed-count state)
                     :max-position (:continuation/max-position state)}))))

(defn start-gpu
  "Create a continuation over an existing resident decoder state.

  Processes every prompt token except the last through ordinary GPU decode, then
  primes the final token into `r0`. `opts` may contain `:model-fingerprint`.
  Throws for an empty prompt or a prompt larger than the decoder's KV capacity."
  [dstate prompt-ids {:keys [model-fingerprint]}]
  (let [{:keys [model maxpos]} dstate
        tokens (mapv long prompt-ids)
        prompt-count (count tokens)
        _ (when (zero? prompt-count)
            (throw (ex-info "A continuation requires a non-empty prompt" {})))
        _ (when (> prompt-count (long maxpos))
            (throw (ex-info "Prompt does not fit in the resident KV cache"
                            {:prompt-count prompt-count :max-position maxpos})))
        processed (dec prompt-count)]
    (doseq [position (range processed)]
      (decoder-gpu/decode-token! dstate (nth tokens position) position))
    (decoder-gpu/prime-resident-token! dstate (peek tokens))
    {:continuation/backend :gpu
     :continuation/dstate dstate
     :continuation/model-fingerprint model-fingerprint
     :continuation/layout (continuation/model-layout model)
     :continuation/max-position (long maxpos)
     :continuation/processed-count processed
     :continuation/pending-token (peek tokens)
     :continuation/tokens tokens}))

(defn step-gpu
  "Advance a GPU continuation by one greedy token.

  Returns `[next-continuation generated-token]`. The recorded decoder graph leaves
  the generated token's embedding resident for the following call."
  [state]
  (when-not (= :gpu (:continuation/backend state))
    (throw (ex-info "step-gpu requires a GPU continuation"
                    {:backend (:continuation/backend state)})))
  (require-room! state)
  (let [position (long (:continuation/processed-count state))
        token (decoder-gpu/resident-step! (:continuation/dstate state) position)]
    [(-> state
         (assoc :continuation/processed-count (inc position)
                :continuation/pending-token token)
         (update :continuation/tokens conj token))
     token]))

(defn advance-gpu
  "Advance a GPU continuation `n` greedy steps.

  Returns `{:continuation state :tokens generated-token-vector}`."
  [state n]
  (loop [state state remaining (long n) tokens []]
    (if (zero? remaining)
      {:continuation state :tokens tokens}
      (let [[next-state token] (step-gpu state)]
        (recur next-state (dec remaining) (conj tokens token))))))

(defn export-gpu-metadata
  "Return the portable non-tensor metadata for a GPU continuation.

  This is used to allocate durable tensor destinations before performing the
  synchronous device-to-host transfer. No GPU data moves in this call."
  [state]
  (when-not (= :gpu (:continuation/backend state))
    (throw (ex-info "export-gpu-metadata requires a GPU continuation"
                    {:backend (:continuation/backend state)})))
  {:continuation/version 1
   :continuation/model-fingerprint (:continuation/model-fingerprint state)
   :continuation/layout (:continuation/layout state)
   :continuation/processed-count (:continuation/processed-count state)
   :continuation/pending-token (:continuation/pending-token state)
   :continuation/tokens (:continuation/tokens state)})

(defn export-gpu-into!
  "Download a GPU continuation's occupied attention-state prefix into `destinations`.

  `destinations` must have the metadata returned by `export-gpu-metadata` and a
  tensor vector for every declared slab, containing one JVM float array or
  writable `MemorySegment` per layer. Raster validates the complete batch.
  Returns `destinations` after the synchronous transfer."
  [state destinations]
  (let [metadata (export-gpu-metadata state)
        dstate (:continuation/dstate state)
        processed (long (:continuation/processed-count state))
        layout (get-in metadata [:continuation/layout :attention-state])
        tensor-keys (mapv :tensor-key (:slabs layout))
        groups (attention-state/tensor-groups destinations)]
    (when-not (= metadata (apply dissoc destinations tensor-keys))
      (throw (ex-info "GPU export destinations do not match continuation metadata"
                      {:expected metadata
                       :actual (apply dissoc destinations tensor-keys)})))
    (when-not (every? (fn [[slab tensors]]
                        (let [elements (* processed (:elements-per-token slab))]
                          (every? #(= elements (host-elements %)) tensors)))
                      groups)
      (throw (ex-info "GPU export destinations do not match the occupied prefix"
                      {:processed-count processed :attention-state layout})))
    (gpu/download-ranges!
     (:sess dstate)
     (vec (for [[slab tensors] groups
                layer (range (:count slab))]
            [(attention-state/buffer-key slab layer) (nth tensors layer)
             {:elements (* processed (:elements-per-token slab))}])))
    destinations))

(defn export-gpu
  "Download only the occupied K/V prefix of a GPU continuation.

  The 2×layers ranged transfers are submitted as one validate-before-copy batch.
  Returns the same portable snapshot shape as `continuation/export-cpu`."
  [state]
  (let [processed (long (:continuation/processed-count state))
        layout (get-in state [:continuation/layout :attention-state])]
    (export-gpu-into!
     state
     (merge (export-gpu-metadata state)
            (allocate-tensor-groups layout processed)))))

(defn export-gpu-chunk
  "Download one immutable token-range descriptor into a contiguous float payload.

  Payload order is all K layers followed by all V layers. Raster validates the
  complete ranged batch before copying. This is the only unavoidable GPU-to-host
  copy; the resulting array can be encoded asynchronously by the cache manager."
  [state descriptor]
  (let [metadata (export-gpu-metadata state)
        dstate (:continuation/dstate state)
        layout (get-in metadata [:continuation/layout :attention-state])
        slabs (attention-state/payload-plan layout (:chunk/token-count descriptor))
        slab-layouts (into {} (map (juxt :name identity)) (:slabs layout))
        payload (float-array (reduce + 0 (map :elements slabs)))]
    (when (> (+ (:chunk/start descriptor) (:chunk/token-count descriptor))
             (:continuation/processed-count state))
      (throw (ex-info "GPU chunk extends beyond the occupied KV prefix"
                      {:descriptor descriptor
                       :processed-count (:continuation/processed-count state)})))
    (gpu/download-ranges!
     (:sess dstate)
     (mapv (fn [{:keys [slab layer element-offset elements]}]
             (let [slab-layout (get slab-layouts slab)]
               [(attention-state/buffer-key slab-layout layer) payload
                {:src-element (* (long (:chunk/start descriptor))
                                 (:elements-per-token slab-layout))
                 :dst-element element-offset
                 :elements elements}]))
           slabs))
    (merge descriptor
           {:chunk/version 2
            :chunk/model-fingerprint (:continuation/model-fingerprint metadata)
            :chunk/layout (:continuation/layout metadata)
            :chunk/slabs slabs
            :chunk/payload payload})))

(defn upload-gpu-chunk!
  "Upload one contiguous chunk payload at its declared token offset.

  `payload` may be a float array or a Konserve mmap MemorySegment. The complete
  per-layer batch is validated before any device buffer is changed."
  [dstate descriptor payload]
  (let [{:keys [model maxpos sess]} dstate
        start (long (:kv/start-token descriptor (:chunk/start descriptor)))
        token-count (long (:kv/token-count descriptor (:chunk/token-count descriptor)))
        layout (attention-state/layout model)
        slabs (attention-state/payload-plan layout token-count)
        slab-layouts (into {} (map (juxt :name identity)) (:slabs layout))
        expected-elements (reduce + 0 (map :elements slabs))]
    (when (> (+ start token-count) (long maxpos))
      (throw (ex-info "KV chunk does not fit in the resident cache"
                      {:start start :token-count token-count :max-position maxpos})))
    (when-not (= expected-elements (host-elements payload))
      (throw (ex-info "KV chunk payload does not match the model layout"
                      {:expected-elements expected-elements
                       :actual-elements (host-elements payload)})))
    (gpu/upload-ranges!
     sess
     (mapv (fn [{:keys [slab layer element-offset elements]}]
             (let [slab-layout (get slab-layouts slab)]
               [(attention-state/buffer-key slab-layout layer) payload
                {:src-element element-offset
                 :dst-element (* start (:elements-per-token slab-layout))
                 :elements elements}]))
           slabs))
    dstate))

(defn resume-prompt-from-prefix
  "Finish a prompt after `[0, cached-token-count)` KV rows were restored.

  Known prompt tokens after the cached prefix are processed normally, and the
  final token is primed as pending. Returns a GPU continuation ready to generate."
  [dstate model-fingerprint tokens cached-token-count]
  (let [tokens (mapv long tokens)
        logical-count (count tokens)
        processed (dec logical-count)
        cached-token-count (long cached-token-count)]
    (when (zero? logical-count)
      (throw (ex-info "A continuation requires a non-empty prompt" {})))
    (when (or (neg? cached-token-count) (> cached-token-count processed))
      (throw (ex-info "Cached token count is outside the prompt KV prefix"
                      {:cached-token-count cached-token-count
                       :processed-count processed})))
    (when (> logical-count (long (:maxpos dstate)))
      (throw (ex-info "Prompt does not fit in the resident KV cache"
                      {:prompt-count logical-count :max-position (:maxpos dstate)})))
    (doseq [position (range cached-token-count processed)]
      (decoder-gpu/decode-token! dstate (nth tokens position) position))
    (decoder-gpu/prime-resident-token! dstate (peek tokens))
    {:continuation/backend :gpu
     :continuation/dstate dstate
     :continuation/model-fingerprint model-fingerprint
     :continuation/layout (continuation/model-layout (:model dstate))
     :continuation/max-position (long (:maxpos dstate))
     :continuation/processed-count processed
     :continuation/pending-token (peek tokens)
     :continuation/tokens tokens}))

(defn restore-gpu
  "Restore a portable continuation snapshot into an existing decoder state.

  All tensor shapes and the complete transfer batch are validated before copying.
  `opts` may contain `:model-fingerprint`; when supplied it must match the snapshot."
  [dstate snapshot {:keys [model-fingerprint]}]
  (let [{:keys [model maxpos sess]} dstate
        expected-layout (continuation/model-layout model)
        snapshot-fingerprint (:continuation/model-fingerprint snapshot)
        processed (long (:continuation/processed-count snapshot))
        layout (:attention-state expected-layout)
        groups (attention-state/tensor-groups snapshot)]
    (when-not (= 1 (:continuation/version snapshot))
      (throw (ex-info "Unsupported continuation snapshot version"
                      {:version (:continuation/version snapshot)})))
    (when-not (= expected-layout (:continuation/layout snapshot))
      (throw (ex-info "Continuation layout does not match the model"
                      {:expected expected-layout :actual (:continuation/layout snapshot)})))
    (when (and snapshot-fingerprint model-fingerprint
               (not= snapshot-fingerprint model-fingerprint))
      (throw (ex-info "Continuation model fingerprint does not match"
                      {:expected snapshot-fingerprint :actual model-fingerprint})))
    (when (> processed (long maxpos))
      (throw (ex-info "Continuation does not fit in the resident KV cache"
                      {:processed-count processed :max-position maxpos})))
    (when-not (every? (fn [[slab tensors]]
                        (let [elements (* processed (:elements-per-token slab))]
                          (every? #(= elements (host-elements %)) tensors)))
                      groups)
      (throw (ex-info "Continuation tensors do not match their declared layout"
                      {:processed-count processed :attention-state layout})))
    (gpu/upload-ranges!
     sess
     (vec (for [[slab tensors] groups
                layer (range (:count slab))]
            [(attention-state/buffer-key slab layer) (nth tensors layer)
             {:elements (* processed (:elements-per-token slab))}])))
    (decoder-gpu/prime-resident-token! dstate (:continuation/pending-token snapshot))
    {:continuation/backend :gpu
     :continuation/dstate dstate
     :continuation/model-fingerprint (or model-fingerprint snapshot-fingerprint)
     :continuation/layout expected-layout
     :continuation/max-position (long maxpos)
     :continuation/processed-count processed
     :continuation/pending-token (:continuation/pending-token snapshot)
     :continuation/tokens (vec (:continuation/tokens snapshot))}))
