(ns pretrained.decision.laya-gpu
  "Fixed-shape resident GPU execution for Laya's decision transformer.

  The encoder and both `torch.nn.TransformerEncoderLayer`-compatible head
  layers are composed before device allocation. Token embedding remains on
  the CPU; question typing is uploaded as a fixed-shape value tensor so the
  resident graph stays on Raster's composable value-map path."
  (:require [pretrained.arch.modernbert :as mb]
            [pretrained.arch.modernbert-gpu :as mb-gpu]
            [raster.arrays :refer [alloc-like]]
            [raster.core :refer [deftm]]
            [raster.dl.array-ops :as ops]
            [raster.dl.attention :as attn]
            [raster.dl.nn :as nn]
            [raster.gpu.compiled :as compiled]
            [raster.gpu.value :as value]
            [raster.numeric :as n]))

(deftm laya-resident-head-body
  (All [T]
       [x :- (Array T)
        norm1-w :- (Array T) norm1-b :- (Array T)
        qkv-w :- (Array T) qkv-b :- (Array T)
        out-w :- (Array T) out-b :- (Array T)
        norm2-w :- (Array T) norm2-b :- (Array T)
        linear1-w :- (Array T) linear1-b :- (Array T)
        linear2-w :- (Array T) linear2-b :- (Array T)
        seq-len :- Long d-model :- Long d-ff :- Long
       n-heads :- Long head-dim :- Long eps :- Double scale :- Double]
       :- (Array T)
       (let [n1 (alloc-like x (* seq-len d-model))
             _ (nn/layer-norm! x norm1-w norm1-b n1 seq-len d-model eps)
             qkv (nn/linear n1 qkv-w qkv-b seq-len d-model (* 3 d-model))
             q (ops/slice-strided-2d qkv seq-len (* 3 d-model) 0 d-model)
             k (ops/slice-strided-2d qkv seq-len (* 3 d-model) d-model d-model)
             v (ops/slice-strided-2d qkv seq-len (* 3 d-model) (* 2 d-model) d-model)
             scores (alloc-like x (* seq-len (* n-heads seq-len)))
             _ (attn/attn-prefill-scores-bidir! q k scores seq-len n-heads 1
                                                 n-heads head-dim scale)
             _ (attn/attn-prefill-softmax! scores seq-len n-heads)
             context (alloc-like x (* seq-len d-model))
             _ (attn/attn-prefill-out! scores v context seq-len n-heads 1
                                       n-heads head-dim)
             projected (nn/linear context out-w out-b seq-len d-model d-model)
             x1 (nn/residual-add x projected (* seq-len d-model))
             n2 (alloc-like x1 (* seq-len d-model))
             _ (nn/layer-norm! x1 norm2-w norm2-b n2 seq-len d-model eps)
             up (nn/linear n2 linear1-w linear1-b seq-len d-model d-ff)
             activated (nn/leaky-relu up (* seq-len d-ff) (n/oftype up 0.0))
             down (nn/linear activated linear2-w linear2-b seq-len d-ff d-model)]
         (nn/residual-add x1 down (* seq-len d-model)))))

(deftm laya-resident-first-head
  (All [T]
       [x :- (Array T) type-values :- (Array T)
        norm1-w :- (Array T) norm1-b :- (Array T)
        qkv-w :- (Array T) qkv-b :- (Array T)
        out-w :- (Array T) out-b :- (Array T)
        norm2-w :- (Array T) norm2-b :- (Array T)
        linear1-w :- (Array T) linear1-b :- (Array T)
        linear2-w :- (Array T) linear2-b :- (Array T)
        seq-len :- Long d-model :- Long d-ff :- Long
        n-heads :- Long head-dim :- Long eps :- Double scale :- Double]
       :- (Array T)
       (let [typed (nn/residual-add x type-values (* seq-len d-model))]
         (laya-resident-head-body
          typed norm1-w norm1-b qkv-w qkv-b out-w out-b norm2-w norm2-b
          linear1-w linear1-b linear2-w linear2-b
          seq-len d-model d-ff n-heads head-dim eps scale))))

(def ^:private first-head-constants
  '[norm1-w norm1-b qkv-w qkv-b out-w out-b norm2-w norm2-b
    linear1-w linear1-b linear2-w linear2-b])

(def ^:private head-constants first-head-constants)

(defn- head-args
  [model x type-embedding seq-len layer]
  (let [d (:d-model model)
        head-dim 64
        heads (quot d head-dim)
        ffn (* 4 d)
        prefix (str "head.layers." layer ".")
        weights [(mb/weight model (str prefix "norm1.weight"))
                 (mb/weight model (str prefix "norm1.bias"))
                 (mb/weight model (str prefix "self_attn.in_proj_weight"))
                 (mb/weight model (str prefix "self_attn.in_proj_bias"))
                 (mb/weight model (str prefix "self_attn.out_proj.weight"))
                 (mb/weight model (str prefix "self_attn.out_proj.bias"))
                 (mb/weight model (str prefix "norm2.weight"))
                 (mb/weight model (str prefix "norm2.bias"))
                 (mb/weight model (str prefix "linear1.weight"))
                 (mb/weight model (str prefix "linear1.bias"))
                 (mb/weight model (str prefix "linear2.weight"))
                 (mb/weight model (str prefix "linear2.bias"))]
        shape [(long seq-len) d ffn heads head-dim 1.0e-5 (/ 1.0 8.0)]]
    (vec (concat (if (zero? layer) [x type-embedding] [x]) weights shape))))

(defn- lower-head
  [model x type-embedding seq-len layer opts]
  (compiled/lower
   (if (zero? layer) #'laya-resident-first-head #'laya-resident-head-body)
   (head-args model x type-embedding seq-len layer)
   (merge {:target :ze:0 :dtype :float
           :constants (if (zero? layer) first-head-constants head-constants)
           :on-non-resident :throw}
          (select-keys opts [:target :gemm-precision :schedule]))))

(defn prepare-decision-hidden
  "Lower and compose a fixed-shape ModernBERT encoder and Laya decision head.
  This is allocation-free and does not contact the GPU."
  ([agent seq-len] (prepare-decision-hidden agent seq-len {}))
  ([agent seq-len opts]
   (let [model (:encoder agent)
         head-layers (long (get (:config agent) "head_layers" 2))
         _ (when-not (= 2 head-layers)
             (throw (ex-info "resident Laya path currently requires two head layers"
                             {:head-layers head-layers})))
         d (:d-model model)
         x (float-array (* (long seq-len) d))
         type-embedding (float-array (* (long seq-len) d))
         encoder (mb-gpu/prepare-encoder model seq-len opts)
         head0 (lower-head model x type-embedding seq-len 0 opts)
         head1 (lower-head model x type-embedding seq-len 1 opts)
         head0-out (-> head0 :out-tree first :key)
         head1-out (-> head1 :out-tree first :key)]
     (compiled/compose
      {:id [:laya-decision-hidden (or (:target opts) :ze:0) seq-len]
       :components [{:id :encoder :program encoder}
                    {:id :head-0 :program head0}
                    {:id :head-1 :program head1}]
       :connections [{:from [:encoder :encoded] :to [:head-0 :x]}
                     {:from [:head-0 head0-out] :to [:head-1 :x]}]
       :outputs [{:key :hidden :from [:head-1 head1-out]}]
       :attributes {:architecture :laya
                    :sequence-length seq-len
                    :head-layers head-layers}}))))

(defn compile-decision-hidden
  "Instantiate and upload a fixed-shape encoder plus decision head."
  ([agent seq-len] (compile-decision-hidden agent seq-len {}))
  ([agent seq-len opts]
   (let [prepared (prepare-decision-hidden agent seq-len opts)]
     {:program (compiled/instantiate! prepared (select-keys opts [:profile?]))
      :prepared prepared
      :embedding-key [:encoder [:layer-0 :x]]
      :type-key [:head-0 :type-values]
      :seq-len (long seq-len)
      :d-model (:d-model (:encoder agent))
      :agent agent})))

(defn question-type-embedding
  "Copy one checkpoint type-embedding row."
  ^floats [agent question-type]
  (let [d (:d-model (:encoder agent))
        all (mb/weight (:encoder agent) "type_emb.weight")
        out (float-array d)]
    (System/arraycopy all (* (long question-type) d) out 0 d)
    out))

(defn- repeated-question-type-values
  ^floats [agent question-type seq-len]
  (let [row (question-type-embedding agent question-type)
        d (alength row)
        out (float-array (* (long seq-len) d))]
    (dotimes [i (long seq-len)]
      (System/arraycopy row 0 out (* i d) d))
    out))

(defn decision-hidden-device
  "Replay a compiled graph for normalized token embeddings and a question type."
  [{:keys [program embedding-key type-key seq-len d-model agent]}
   embeddings question-type]
  (when-not (= (* seq-len d-model) (alength embeddings))
    (throw (ex-info "Laya GPU embedding shape mismatch"
                    {:expected (* seq-len d-model) :actual (alength embeddings)})))
  (:hidden (program {embedding-key embeddings
                     type-key (repeated-question-type-values agent question-type seq-len)})))

(defn decision-hidden
  "Replay a compiled graph and download the final decision hidden states."
  [compiled embeddings question-type]
  (value/->host (decision-hidden-device compiled embeddings question-type)))

(defn embed-token-ids
  "Perform Laya's inexpensive token lookup and embedding normalization on CPU."
  ^floats [{:keys [agent seq-len d-model]} token-ids]
  (when-not (= seq-len (count token-ids))
    (throw (ex-info "token sequence does not match compiled Laya GPU shape"
                    {:compiled-length seq-len :actual-length (count token-ids)})))
  (let [model (:encoder agent)
        ids (long-array token-ids)
        embedded (nn/embedding
                  (mb/weight model "encoder.embeddings.tok_embeddings.weight")
                  ids seq-len (:vocab-size model) d-model)]
    (nn/layer-norm embedded
                   (mb/weight model "encoder.embeddings.norm.weight")
                   (:zero-bias model) seq-len d-model (:eps model))))

(defn decision-hidden-tokens
  "CPU-embed tokens, replay the resident encoder/head, and download hidden states."
  [compiled token-ids question-type]
  (decision-hidden compiled (embed-token-ids compiled token-ids) question-type))

(defn close! [{:keys [program]}]
  (compiled/close! program))
