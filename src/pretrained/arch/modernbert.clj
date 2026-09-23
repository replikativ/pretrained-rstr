(ns pretrained.arch.modernbert
  "ModernBERT encoder inference for Laya checkpoints.

  This is the correctness-first CPU path: one unpadded sequence, F16 checkpoint
  weights expanded to F32, BLAS linears, and the same global/local attention and
  exact GELU as the Transformers reference. Batching and resident quantized GPU
  execution build on this numerical anchor."
  (:require [clojure.data.json :as json]
            [raster.arrays :refer [alloc-like]]
            [raster.core :refer [deftm]]
            [raster.dl.array-ops :as ops]
            [raster.dl.attention :as attn]
            [raster.dl.nn :as nn]
            [raster.linalg.blas :as blas]
            [raster.par]
            [pretrained.safetensors :as st]))

(deftm modernbert-block
  (All [T]
       [x :- (Array T)
        attn-norm :- (Array T) mlp-norm :- (Array T) zero-bias :- (Array T)
        wqkv :- (Array T) wo :- (Array T) wi :- (Array T) w-mlp-out :- (Array T)
        seq-len :- Long d-model :- Long d-ff :- Long n-heads :- Long head-dim :- Long
        theta :- Double scale :- Double eps :- Double window :- Long normalize-attn :- Long]
       :- (Array T)
       (let [attn-in (if (zero? normalize-attn)
                       x
                       (nn/layer-norm x attn-norm zero-bias seq-len d-model eps))
             qkv (nn/linear-nb attn-in wqkv seq-len d-model (* 3 d-model))
             q (ops/slice-strided-2d qkv seq-len (* 3 d-model) 0 d-model)
             k (ops/slice-strided-2d qkv seq-len (* 3 d-model) d-model d-model)
             v (ops/slice-strided-2d qkv seq-len (* 3 d-model) (* 2 d-model) d-model)
             q (attn/rope-pos q seq-len n-heads head-dim theta 0)
             k (attn/rope-pos k seq-len n-heads head-dim theta 0)
             scores (alloc-like x (* seq-len (* n-heads seq-len)))
             _ (if (zero? window)
                 (attn/attn-prefill-scores-bidir! q k scores seq-len n-heads 1
                                                  n-heads head-dim scale)
                 (attn/attn-prefill-scores-windowed! q k scores seq-len n-heads 1
                                                     n-heads head-dim scale window window))
             _ (attn/attn-prefill-softmax! scores seq-len n-heads)
             context (alloc-like x (* seq-len d-model))
             _ (attn/attn-prefill-out! scores v context seq-len n-heads 1
                                       n-heads head-dim)
             projected (nn/linear-nb context wo seq-len d-model d-model)
             x1 (nn/residual-add x projected (* seq-len d-model))
             mlp-in (nn/layer-norm x1 mlp-norm zero-bias seq-len d-model eps)
             fused (nn/linear-nb mlp-in wi seq-len d-model (* 2 d-ff))
             activated (ops/slice-strided-2d fused seq-len (* 2 d-ff) 0 d-ff)
             gate (ops/slice-strided-2d fused seq-len (* 2 d-ff) d-ff d-ff)
             _ (nn/gelu-erf! activated activated (* seq-len d-ff))
             hidden (nn/hadamard activated gate (* seq-len d-ff))
             down (nn/linear-nb hidden w-mlp-out seq-len d-ff d-model)]
         (nn/residual-add x1 down (* seq-len d-model)))))

;; Resident GPU specialization. Host conditionals are deliberately outside this
;; graph: full attention is the exact windowed case window=seq-len, while local
;; layers pass their configured radius. This leaves a flat kernel/GEMM sequence
;; for Raster's resident extractor.
(deftm modernbert-resident-body
  (All [T]
       [x :- (Array T) attn-in :- (Array T)
        mlp-norm :- (Array T) zero-bias :- (Array T)
        wqkv :- (Array T) wo :- (Array T) wi :- (Array T) w-mlp-out :- (Array T)
        seq-len :- Long d-model :- Long d-ff :- Long n-heads :- Long head-dim :- Long
        theta :- Double scale :- Double eps :- Double window :- Long]
       :- (Array T)
       (let [qkv (nn/linear-nb attn-in wqkv seq-len d-model (* 3 d-model))
             qr (alloc-like qkv (* seq-len d-model))
             kr (alloc-like qkv (* seq-len d-model))
             _ (raster.dl.attention/rope-prefill-strided!
                qkv qr seq-len n-heads head-dim theta (* 3 d-model) 0)
             _ (raster.dl.attention/rope-prefill-strided!
                qkv kr seq-len n-heads head-dim theta (* 3 d-model) d-model)
             scores (alloc-like x (* seq-len (* n-heads seq-len)))
             _ (attn/attn-prefill-scores-windowed! qr kr scores seq-len n-heads 1
                                                   n-heads head-dim scale window window)
             probs (raster.dl.attention/attn-prefill-softmax scores seq-len n-heads)
             context (alloc-like x (* seq-len d-model))
             _ (raster.dl.attention/attn-prefill-out-strided!
                probs qkv context seq-len n-heads 1 n-heads head-dim
                (* 3 d-model) (* 2 d-model))
             projected (nn/linear-nb context wo seq-len d-model d-model)
             x1-buffer (alloc-like x (* seq-len d-model))
             x1 (raster.par/map! x1-buffer i (* seq-len d-model) nil
                                 (+ (aget x i) (aget projected i)))
             mlp-in (raster.dl.nn/layer-norm-reassociated
                     x1 mlp-norm zero-bias seq-len d-model eps)
             fused (nn/linear-nb mlp-in wi seq-len d-model (* 2 d-ff))
             hidden (alloc-like fused (* seq-len d-ff))
             _ (raster.dl.nn/gelu-erf-mul-strided!
                fused hidden seq-len (* 2 d-ff) 0 d-ff d-ff)
             down (nn/linear-nb hidden w-mlp-out seq-len d-ff d-model)
             out-buffer (alloc-like x (* seq-len d-model))]
         (raster.par/map! out-buffer i (* seq-len d-model) nil
                          (+ (aget x1 i) (aget down i))))))

(deftm modernbert-resident-first-block
  (All [T]
       [x :- (Array T) mlp-norm :- (Array T) zero-bias :- (Array T)
        wqkv :- (Array T) wo :- (Array T) wi :- (Array T) w-mlp-out :- (Array T)
        seq-len :- Long d-model :- Long d-ff :- Long n-heads :- Long head-dim :- Long
        theta :- Double scale :- Double eps :- Double window :- Long]
       :- (Array T)
       (modernbert-resident-body x x mlp-norm zero-bias wqkv wo wi w-mlp-out
                                 seq-len d-model d-ff n-heads head-dim
                                 theta scale eps window)))

(deftm modernbert-resident-block
  (All [T]
       [x :- (Array T) attn-norm :- (Array T) mlp-norm :- (Array T)
        zero-bias :- (Array T)
        wqkv :- (Array T) wo :- (Array T) wi :- (Array T) w-mlp-out :- (Array T)
        seq-len :- Long d-model :- Long d-ff :- Long n-heads :- Long head-dim :- Long
        theta :- Double scale :- Double eps :- Double window :- Long]
       :- (Array T)
  (let [attn-in (raster.dl.nn/layer-norm-reassociated
                 x attn-norm zero-bias seq-len d-model eps)]
         (modernbert-resident-body x attn-in mlp-norm zero-bias wqkv wo wi w-mlp-out
                                   seq-len d-model d-ff n-heads head-dim
                                   theta scale eps window))))

(defn- native-kernel [v]
  (delay
    (try
      ((requiring-resolve 'raster.compiler.pipeline/compile-aot)
       v :dtype :float :target :c :simd? false)
      (catch Throwable error
        (binding [*out* *err*]
          (println "Native ModernBERT kernel compilation failed; using JVM:"
                   (:name (meta v)) "-" (.getMessage error)))
        @v))))

(def ^:private native-gelu (native-kernel #'nn/gelu-erf!))
(def ^:private native-layer-norm
  (native-kernel #'nn/layer-norm-reassociated!))
(def ^:private native-gelu-mul-strided
  (native-kernel #'nn/gelu-erf-mul-strided!))
(def ^:private native-relu (native-kernel #'nn/leaky-relu!))
(def ^:private native-rope-strided
  (native-kernel #'attn/rope-prefill-strided!))
(def ^:private native-rope-strided-table
  (native-kernel #'attn/rope-prefill-strided-table!))
(def ^:private native-pack-heads-strided
  (native-kernel #'ops/pack-heads-strided))
(def ^:private native-segmented-mask
  (native-kernel #'attn/attn-prefill-mask-segmented-head-major!))
(def ^:private native-softmax (native-kernel #'attn/attn-prefill-softmax!))

(defn- window-mask-fallback!
  [^floats scores nrows heads left right]
  (dotimes [head (long heads)]
    (dotimes [i (long nrows)]
      (dotimes [j (long nrows)]
        (when (or (> (- i j) (dec (long left)))
                  (and (< i j) (> (- j i) (dec (long right)))))
          (aset scores (+ (* head nrows nrows) (* i nrows) j) (float -1.0e30)))))))

;; Keep pretrained-rstr source-compatible with the last released Raster while
;; the new primitive is exercised through :dev's sibling checkout.
(def ^:private native-window-mask
  (if-let [v (ns-resolve 'raster.dl.attention
                         'attn-prefill-mask-windowed-head-major!)]
    (native-kernel v)
    (delay window-mask-fallback!)))

(defn gelu-erf!
  ^floats [^floats x ^floats out n]
  (@native-gelu x out (long n))
  out)

(defn layer-norm
  "Compiled float LayerNorm into caller-independent storage. Raster's
  reassociation contract permits the backend to schedule each row's reductions
  while retaining the centered two-pass formula."
  ^floats [^floats x ^floats gamma ^floats beta rows features eps]
  (let [out (float-array (* (long rows) (long features)))]
    (@native-layer-norm x gamma beta out (long rows) (long features) (double eps))
    out))

(defn relu!
  ^floats [^floats x ^floats out n]
  (@native-relu x out (long n) (float 0.0))
  out)

(defn gelu-erf-mul-strided!
  ^floats [^floats src ^floats out rows row-stride left-offset right-offset width]
  (@native-gelu-mul-strided src out (long rows) (long row-stride)
                           (long left-offset) (long right-offset) (long width))
  out)

(defn rope-strided!
  ^floats [^floats src ^floats out nrows heads head-dim theta row-stride column-offset]
  (@native-rope-strided src out (long nrows) (long heads) (long head-dim)
                        (double theta) (long row-stride) (long column-offset))
  out)

(defn rope-strided-table!
  ^floats [^floats src ^floats cosines ^floats sines ^floats out
           nrows heads head-dim row-stride column-offset]
  (@native-rope-strided-table src cosines sines out
                              (long nrows) (long heads) (long head-dim)
                              (long row-stride) (long column-offset))
  out)

(defn- pack-heads-strided
  ^floats [^floats src nrows heads head-dim row-stride column-offset]
  ;; AOT value functions reuse their result workspace. Clone because attention
  ;; retains Q, K, and V packs simultaneously for the two BLAS contractions.
  (aclone ^floats (@native-pack-heads-strided
                   src (long nrows) (long heads) (long head-dim)
                   (long row-stride) (long column-offset))))

(defn attention-strided!
  "Attention over Q/K/V fields embedded in independently strided row-major
  sources. Packing remains a generic layout map and both contractions remain
  true strided-batch BLAS calls."
  ^floats [^floats q ^floats k ^floats v ^floats scores ^floats out
           nrows heads head-dim scale window
           q-row-stride q-column-offset
           k-row-stride k-column-offset
           v-row-stride v-column-offset]
  (let [qh (pack-heads-strided q nrows heads head-dim q-row-stride q-column-offset)
        kh (pack-heads-strided k nrows heads head-dim k-row-stride k-column-offset)
        vh (pack-heads-strided v nrows heads head-dim v-row-stride v-column-offset)
        context-h (float-array (* (long heads) (long nrows) (long head-dim)))]
    (blas/batched-gemm-nt! qh kh scores (long heads) (long nrows)
                           (long head-dim) (long nrows) (float scale))
    (when (pos? (long window))
      (@native-window-mask scores (long nrows) (long heads)
                           (long window) (long window)))
    (@native-softmax scores (long nrows) (long heads))
    (blas/batched-gemm-nn! scores vh context-h (long heads) (long nrows)
                           (long nrows) (long head-dim) (float 1.0))
    (let [context (ops/unpack-heads context-h nrows heads head-dim)]
      (System/arraycopy context 0 out 0 (alength out))))
  out)

(defn attention-segmented-strided!
  "Uniform padded batch of independent attention segments. Q/K/V use the
  ordinary row-strided view contract; `lengths` prevents padding or neighboring
  examples from entering any active softmax row. Matrix batches are ordered
  head-major then example-major, matching `pack-heads-strided`."
  ^floats [^floats q ^floats k ^floats v ^floats out lengths
           batch nrows heads head-dim scale left right
           q-row-stride q-column-offset
           k-row-stride k-column-offset
           v-row-stride v-column-offset]
  (let [rows (* (long batch) (long nrows))
        matrix-batch (* (long heads) (long batch))
        ^longs lengths (long-array lengths)
        qh (pack-heads-strided q rows heads head-dim q-row-stride q-column-offset)
        kh (pack-heads-strided k rows heads head-dim k-row-stride k-column-offset)
        vh (pack-heads-strided v rows heads head-dim v-row-stride v-column-offset)
        scores (float-array (* matrix-batch (long nrows) (long nrows)))
        context-h (float-array (* matrix-batch (long nrows) (long head-dim)))]
    (blas/batched-gemm-nt! qh kh scores matrix-batch (long nrows)
                           (long head-dim) (long nrows) (float scale))
    (@native-segmented-mask scores lengths (long batch) (long nrows) (long heads)
                            (long left) (long right))
    (@native-softmax scores (long nrows) matrix-batch)
    (blas/batched-gemm-nn! scores vh context-h matrix-batch (long nrows)
                           (long nrows) (long head-dim) (float 1.0))
    (let [context (ops/unpack-heads context-h rows heads head-dim)]
      (System/arraycopy context 0 out 0 (alength out))))
  out)

(defn attention!
  "Compute one full or symmetric-window attention into caller-owned `out`.
  Heads are packed once, then QK^T and probabilities@V use Raster's true
  strided-batch BLAS primitives. The exact softmax stays in native scalar C;
  this avoids oversubscribing MKL's worker pool with a second OpenMP runtime."
  ^floats [^floats q ^floats k ^floats v ^floats scores ^floats out
           nrows heads head-dim scale window]
  (let [width (* (long heads) (long head-dim))]
    (attention-strided! q k v scores out nrows heads head-dim scale window
                        width 0 width 0 width 0)))

(def ^:private compiled-block
  (delay
    (try
      ((requiring-resolve 'raster.compiler.pipeline/compile-aot)
       #'modernbert-block :dtype :float)
      (catch Throwable error
        (binding [*out* *err*]
          (println "ModernBERT AOT compilation failed; using interpreted block:"
                   (.getMessage error)))
        modernbert-block))))

(defn- rope-table
  [nrows head-dim theta]
  (let [half (quot (long head-dim) 2)
        cosines (float-array (* (long nrows) half))
        sines (float-array (* (long nrows) half))
        log-theta (Math/log (double theta))]
    (dotimes [row (long nrows)]
      (dotimes [i half]
        (let [frequency (Math/exp (* (/ (* -2.0 i) (double head-dim)) log-theta))
              angle (* (double row) frequency)
              index (+ (* row half) i)]
          (aset cosines index (float (Math/cos angle)))
          (aset sines index (float (Math/sin angle))))))
    {:cosines cosines :sines sines}))

(defn load-model
  "Load the ModernBERT encoder embedded in a Laya checkpoint directory."
  [dir]
  (let [cfg (json/read-str (slurp (str dir "/encoder/config.json")) :key-fn keyword)
        d (long (:hidden_size cfg))
        n-heads (long (:num_attention_heads cfg))
        head-dim (quot d n-heads)
        max-position (long (or (:max_position_embeddings cfg) 8192))
        global-theta (double (or (:global_rope_theta cfg) 160000.0))
        local-theta (double (or (:local_rope_theta cfg) 10000.0))]
    {:dir dir
     :config cfg
     :weights (st/load-safetensors (str dir "/model.safetensors"))
     :d-model d
     :d-ff (long (:intermediate_size cfg))
     :n-layers (long (:num_hidden_layers cfg))
     :n-heads n-heads
     :head-dim head-dim
     :vocab-size (long (:vocab_size cfg))
     :eps (double (or (:norm_eps cfg) (:layer_norm_eps cfg) 1.0e-5))
     :local-attention (long (:local_attention cfg))
     :global-every (long (:global_attn_every_n_layers cfg))
     ;; Transformers' ModernBertConfig consumes the legacy top-level theta
     ;; fields. Some converted mmBERT configs also contain `rope_parameters`,
     ;; but the reference model currently ignores that map; notably its nested
     ;; sliding theta is 160000 while the executed local_rope_theta is 10000.
     :global-theta global-theta
     :local-theta local-theta
     :rope-tables (into {}
                        (map (fn [theta]
                               [theta (rope-table max-position head-dim theta)]))
                        (distinct [global-theta local-theta]))
     :zero-bias (float-array d)}))

(defn weight
  "Required tensor data, failing with the exact missing checkpoint key."
  ^floats [model name]
  (or (:data (get (:weights model) name))
      (throw (ex-info (str "ModernBERT checkpoint tensor missing: " name)
                      {:tensor name :dir (:dir model)}))))

(defn- layer-norm-nb ^floats [model ^floats x rows ^floats gamma]
  (layer-norm x gamma (:zero-bias model) rows (:d-model model) (:eps model)))

(defn- residual ^floats [^floats a ^floats b]
  (nn/residual-add a b (long (alength a))))

(defn- attention
  ^floats [model ^floats x seq-len layer]
  (let [{:keys [d-model n-heads head-dim global-every local-attention]} model
        prefix (str "encoder.layers." layer ".")
        input (if (zero? layer)
                x
                (layer-norm-nb model x seq-len
                               (weight model (str prefix "attn_norm.weight"))))
        qkv (nn/linear-nb input (weight model (str prefix "attn.Wqkv.weight"))
                          seq-len d-model (* 3 d-model))
        q (ops/slice-strided-2d qkv seq-len (* 3 d-model) 0 d-model)
        k (ops/slice-strided-2d qkv seq-len (* 3 d-model) d-model d-model)
        v (ops/slice-strided-2d qkv seq-len (* 3 d-model) (* 2 d-model) d-model)
        global? (zero? (mod layer global-every))
        theta (if global? (:global-theta model) (:local-theta model))
        q (attn/rope-pos q seq-len n-heads head-dim theta 0)
        k (attn/rope-pos k seq-len n-heads head-dim theta 0)
        scores (float-array (* seq-len n-heads seq-len))
        scale (/ 1.0 (Math/sqrt (double head-dim)))
        context (float-array (* seq-len d-model))
        _ (attention! q k v scores context seq-len n-heads head-dim scale
                      (if global? 0 (inc (quot local-attention 2))))
        projected (nn/linear-nb context (weight model (str prefix "attn.Wo.weight"))
                                seq-len d-model d-model)]
    (residual x projected)))

(defn- mlp
  ^floats [model ^floats x seq-len layer]
  (let [{:keys [d-model d-ff]} model
        prefix (str "encoder.layers." layer ".")
        input (layer-norm-nb model x seq-len
                             (weight model (str prefix "mlp_norm.weight")))
        fused (nn/linear-nb input (weight model (str prefix "mlp.Wi.weight"))
                            seq-len d-model (* 2 d-ff))
        activated (ops/slice-strided-2d fused seq-len (* 2 d-ff) 0 d-ff)
        gate (ops/slice-strided-2d fused seq-len (* 2 d-ff) d-ff d-ff)
        _ (gelu-erf! activated activated (* seq-len d-ff))
        hidden (nn/hadamard activated gate (* seq-len d-ff))
        projected (nn/linear-nb hidden (weight model (str prefix "mlp.Wo.weight"))
                                seq-len d-ff d-model)]
    (residual x projected)))

(defn- run-block
  ^floats [model ^floats x seq-len layer]
  (let [{:keys [d-model d-ff n-heads head-dim global-every local-attention
                global-theta local-theta eps zero-bias]} model
        prefix (str "encoder.layers." layer ".")
        global? (zero? (mod layer global-every))
        ;; Layer zero consumes the already-normalized embeddings directly; the
        ;; checkpoint consequently has no encoder.layers.0.attn_norm tensor.
        attn-norm (if (zero? layer)
                    zero-bias
                    (weight model (str prefix "attn_norm.weight")))]
    ;; AOT functions own and reuse their output workspace.  Clone at the layer
    ;; boundary so the next invocation cannot clear the residual stream while
    ;; preparing its output buffer.
    (aclone
     ^floats
     (@compiled-block
      x attn-norm (weight model (str prefix "mlp_norm.weight")) zero-bias
      (weight model (str prefix "attn.Wqkv.weight"))
      (weight model (str prefix "attn.Wo.weight"))
      (weight model (str prefix "mlp.Wi.weight"))
      (weight model (str prefix "mlp.Wo.weight"))
      seq-len d-model d-ff n-heads head-dim
      (if global? global-theta local-theta)
      (/ 1.0 (Math/sqrt (double head-dim))) eps
      (if global? 0 (inc (quot local-attention 2)))
      (if (zero? layer) 0 1)))))

(declare encode-batch)

(defn encode
  "Encode one unpadded token-id sequence. Optional `observe` receives
  `[point layer array]` for :embeddings, :attention, :layer and :final, allowing
  reference parity without changing the execution path."
  ([model token-ids]
   (let [{:keys [data]} (encode-batch model [token-ids])]
     data))
  ([model token-ids observe]
   (let [{:keys [d-model vocab-size n-layers]} model
         ids (long-array token-ids)
         seq-len (long (alength ids))
         embedded (nn/embedding (weight model "encoder.embeddings.tok_embeddings.weight")
                                ids seq-len vocab-size d-model)
         x0 (layer-norm-nb model embedded seq-len
                           (weight model "encoder.embeddings.norm.weight"))]
     (when observe (observe :embeddings -1 x0))
     (loop [x x0 layer 0]
       (if (< layer n-layers)
         (let [a (when observe (attention model x seq-len layer))
               _ (when observe (observe :attention layer a))
               y (if observe (mlp model a seq-len layer)
                     (run-block model x seq-len layer))
               _ (when observe (observe :layer layer y))]
           (recur y (inc layer)))
         (let [out (layer-norm-nb model x seq-len
                                  (weight model "encoder.final_norm.weight"))]
           (when observe (observe :final n-layers out))
           out))))))

(defn- attention-batch
  ^floats [model ^floats x batch max-len lengths layer]
  (let [{:keys [d-model n-heads head-dim global-every local-attention]} model
        rows (* batch max-len)
        prefix (str "encoder.layers." layer ".")
        input (if (zero? layer)
                x
                (layer-norm-nb model x rows
                               (weight model (str prefix "attn_norm.weight"))))
        qkv (nn/linear-nb input (weight model (str prefix "attn.Wqkv.weight"))
                          rows d-model (* 3 d-model))
        global? (zero? (mod layer global-every))
        theta (if global? (:global-theta model) (:local-theta model))
        {:keys [cosines sines]} (get (:rope-tables model) theta)
        scale (/ 1.0 (Math/sqrt (double head-dim)))
        context (float-array (* rows d-model))]
    ;; Keep attention segmented by example. Linears above and below operate on
    ;; the whole padded batch, while these copies preserve exact single-example
    ;; RoPE positions and ensure no active row can see padding or another item.
    (doseq [b (range batch)]
      (let [len (long (nth lengths b))
            row0 (* b max-len)
            source-row0 (* row0 (* 3 d-model))
            qb (float-array (* len d-model))
            kb (float-array (* len d-model))
            _ (rope-strided-table! qkv cosines sines qb len n-heads head-dim
                                   (* 3 d-model) source-row0)
            _ (rope-strided-table! qkv cosines sines kb len n-heads head-dim
                                   (* 3 d-model) (+ source-row0 d-model))
            scores (float-array (* len n-heads len))
            out (float-array (* len d-model))
            _ (attention-strided!
               qb kb qkv scores out len n-heads head-dim scale
               (if global? 0 (inc (quot local-attention 2)))
               d-model 0 d-model 0 (* 3 d-model) (+ source-row0 (* 2 d-model)))]
        (System/arraycopy out 0 context (* row0 d-model) (* len d-model))))
    (residual x (nn/linear-nb context (weight model (str prefix "attn.Wo.weight"))
                              rows d-model d-model))))

(defn- mlp-batch
  ^floats [model ^floats x rows layer]
  (let [{:keys [d-model d-ff]} model
        prefix (str "encoder.layers." layer ".")
        input (layer-norm-nb model x rows (weight model (str prefix "mlp_norm.weight")))
        fused (nn/linear-nb input (weight model (str prefix "mlp.Wi.weight"))
                            rows d-model (* 2 d-ff))
        hidden (float-array (* rows d-ff))
        _ (gelu-erf-mul-strided! fused hidden rows (* 2 d-ff) 0 d-ff d-ff)]
    (residual x (nn/linear-nb hidden (weight model (str prefix "mlp.Wo.weight"))
                              rows d-ff d-model))))

(defn encode-batch
  "Encode variable-length token sequences in one padded set of encoder GEMMs.
  Attention remains explicitly segmented by item, so padding and neighboring
  questions cannot affect active rows. Returns `{:data :lengths :max-len}` with
  data shaped `[batch,max-len,d-model]`."
  [model token-sequences]
  (when (empty? token-sequences)
    (throw (ex-info "ModernBERT batch must contain at least one sequence" {})))
  (let [{:keys [d-model vocab-size n-layers]} model
        sequences (mapv vec token-sequences)
        lengths (mapv count sequences)
        batch (count sequences)
        max-len (long (reduce max lengths))
        rows (* batch max-len)
        pad-id (long (or (get-in model [:config :pad_token_id]) 0))
        ids (long-array rows pad-id)]
    (doseq [[b sequence] (map-indexed vector sequences)
            [i token] (map-indexed vector sequence)]
      (aset ids (+ (* b max-len) i) (long token)))
    (let [embedded (nn/embedding (weight model "encoder.embeddings.tok_embeddings.weight")
                                 ids rows vocab-size d-model)
          x0 (layer-norm-nb model embedded rows
                            (weight model "encoder.embeddings.norm.weight"))
          encoded (loop [x x0 layer 0]
                    (if (< layer n-layers)
                      (recur (mlp-batch model
                                        (attention-batch model x batch max-len lengths layer)
                                        rows layer)
                             (inc layer))
                      x))]
      {:data (layer-norm-nb model encoded rows
                            (weight model "encoder.final_norm.weight"))
       :lengths lengths :max-len max-len :batch batch})))
