(ns pretrained.arch.bert
  "BERT encoder inference using pre-trained weights from SafeTensors.

  The encoder counterpart to the descriptor-driven decoder engine: an ordinary
  transformer *encoder* (not a causal decoder), so it does not fit the decoder
  descriptor — it is a self-contained architecture used by the
  sentence-transformers MiniLM/MPNet/BGE family (BERT encoder + mean pool + L2).
  embed.clj routes `:engine :encoder` registry entries here.

  Implements the standard BERT architecture:
    Embeddings (token + position + type) → N × TransformerBlock → pool

  Each TransformerBlock:
    x → MHA(Q,K,V,O) → add+LN → FFN(up,down) → add+LN

  Composes parametric ops from raster's dl/nn and dl/attention substrate for
  type-generic inference (float[] or double[]). Weight layout follows
  HuggingFace naming conventions."
  (:refer-clojure :exclude [aget aset alength aclone + - * /])
  (:require [raster.core :refer [deftm]]
            [raster.arrays :refer [aget aset alength]]
            [raster.numeric :as n :refer [+ - * /]]
            [raster.dl.nn :as nn]
            [raster.dl.attention :as attn]
            [pretrained.safetensors :as st]
            [pretrained.tokenizer.wordpiece :as tok]
            [clojure.data.json :as json]))

;; ================================================================
;; BERT Embeddings: token + position + token_type → LayerNorm
;; ================================================================

(deftm bert-embeddings (All [T]
  [token-emb :- (Array T) pos-emb :- (Array T) type-emb :- (Array T)
   ln-gamma :- (Array T) ln-beta :- (Array T)
   seq-len :- Long d-model :- Long eps :- Double]
  :- (Array T)
  (let [_ (dotimes [i (* seq-len d-model)]
            (aset token-emb i (+ (aget token-emb i)
                                 (aget pos-emb i)
                                 (aget type-emb i))))]
    (nn/layer-norm token-emb ln-gamma ln-beta seq-len d-model eps))))

;; ================================================================
;; Encoder block (deftm — 20 params, at IFn limit)
;; ================================================================

(deftm bert-block (All [T]
  [x :- (Array T)
   wq :- (Array T) bq :- (Array T) wk :- (Array T) bk :- (Array T)
   wv :- (Array T) bv :- (Array T) wo :- (Array T) bo :- (Array T)
   attn-ln-w :- (Array T) attn-ln-b :- (Array T)
   fc-w :- (Array T) fc-b :- (Array T)
   proj-w :- (Array T) proj-b :- (Array T)
   ffn-ln-w :- (Array T) ffn-ln-b :- (Array T)
   seq-len :- Long hidden-size :- Long num-heads :- Long]
  :- (Array T)
  (let [n (* seq-len hidden-size)
        intermediate-size (* 4 hidden-size)
        attn-out (attn/multi-head-attention
                  x wq bq wk bk wv bv wo bo
                  seq-len hidden-size num-heads)
        ;; fused residual-add + layer-norm (SkipLayerNorm) — one kernel each,
        ;; eliminating the two residual-add passes + their intermediate buffers.
        x1 (nn/skip-layer-norm attn-out x attn-ln-w attn-ln-b
                               seq-len hidden-size 1e-12)
        ffn-up (nn/linear x1 fc-w fc-b seq-len hidden-size intermediate-size)
        ffn-up (nn/gelu ffn-up (* seq-len intermediate-size))
        ffn-down (nn/linear ffn-up proj-w proj-b
                            seq-len intermediate-size hidden-size)]
    (nn/skip-layer-norm ffn-down x1 ffn-ln-w ffn-ln-b
                        seq-len hidden-size 1e-12))))

;; ================================================================
;; Mean pooling + L2 normalize: raster.dl.nn substrate ops
;; (nn/mean-pool + nn/l2-normalize!) — see sentence-embedding below.
;; ================================================================

(deftm cosine-similarity (All [T]
  [a :- (Array T) b :- (Array T) n :- Long]
  :- Double
  (loop [i 0 acc 0.0]
    (if (< i n)
      (recur (+ i 1) (+ acc (* (aget a i) (aget b i))))
      acc))))

;; ================================================================
;; Model loading + forward (plain defn — pulls weights from a map)
;; ================================================================

(defn load-model
  "Load a BERT model from a directory containing model.safetensors and
  config.json. Returns a model map with :weights, :config, and hyperparameters."
  [dir]
  (let [config (json/read-str (slurp (str dir "/config.json")) :key-fn keyword)
        weights (st/load-safetensors (str dir "/model.safetensors"))]
    {:weights weights
     :config config
     :hidden-size (:hidden_size config)
     :num-layers (:num_hidden_layers config)
     :num-heads (:num_attention_heads config)
     :intermediate-size (:intermediate_size config)
     :vocab-size (:vocab_size config)
     :max-position (:max_position_embeddings config)
     :layer-norm-eps (double (:layer_norm_eps config 1e-12))}))

;; ================================================================
;; Compiled encoder block
;; ================================================================

(def ^:private compiled-blocks
  "dtype -> delay of the compile-aot'd `bert-block`.

  Invoked directly, a deftm runs raster's default tier, where par forms expand
  to sequential loops. `attention/softmax-rows!` is written for the opposite
  assumption: its exp is a flat `broadcast` carrying an inlined deg-6 Taylor +
  10 squarings (~26 float ops/element) because that is nearly free once the
  broadcast becomes SIMD lanes. Unvectorized, the same shape is a scalar loop
  over every element and the optimization inverts — measured at seq-len 128 /
  hidden 384 / 12 heads:

    softmax-rows!   758.8 ms  ->    0.83 ms   (911x)
    bert-block      787.0 ms  ->   13.3 ms    (59x)
    embed (MiniLM)  714.7 ms  ->   40.2 ms    (17.8x)

  compile-aot vectorizes it (`{:simd-maps 1, :fallback 0}`); the GEMMs already
  reach MKL either way, so this is the whole gap. Numerics are unchanged to f32
  reassociation: cosine 1.0000005, max abs diff 3.4e-7.

  One compile per dtype amortizes over every sequence the model embeds."
  (atom {}))

(defn- block-fn
  "The compiled `bert-block` for `dtype`, falling back to the deftm itself if
  compile-aot fails — backends are WIP, and a slow encode beats no encode."
  [dtype]
  @(get (swap! compiled-blocks
               (fn [m]
                 (if (contains? m dtype)
                   m
                   (assoc m dtype
                          (delay
                            (try
                              ((requiring-resolve
                                'raster.compiler.pipeline/compile-aot)
                               #'bert-block :dtype dtype)
                              (catch Throwable _ bert-block)))))))
        dtype))

(def ^:private float-array-class (Class/forName "[F"))

(defn- array-dtype
  "compile-aot selects an overload by dtype; pick it from the actual weights
  rather than the config, since safetensors decides f32 vs f64."
  [arr]
  (if (instance? float-array-class arr) :float :double))

(defn encode
  "Run the BERT encoder forward pass for one sequence.
  model: from load-model. token-ids: long[].
  Returns hidden states [seq_len, hidden_size] as float[]/double[]."
  [model ^longs token-ids]
  (let [{:keys [weights hidden-size num-layers num-heads
                vocab-size max-position layer-norm-eps]} model
        seq-len (clojure.core/alength token-ids)
        w (fn [^String nm] (:data (get weights nm)))
        dim (long hidden-size)
        tok-emb (nn/embedding (w "embeddings.word_embeddings.weight")
                              token-ids seq-len (long vocab-size) dim)
        pos-ids (long-array seq-len)
        _ (dotimes [i seq-len] (clojure.core/aset pos-ids i (long i)))
        pos-emb (nn/embedding (w "embeddings.position_embeddings.weight")
                              pos-ids seq-len (long max-position) dim)
        type-ids (long-array seq-len)
        type-emb (nn/embedding (w "embeddings.token_type_embeddings.weight")
                               type-ids seq-len 2 dim)
        x (bert-embeddings tok-emb pos-emb type-emb
                           (w "embeddings.LayerNorm.weight")
                           (w "embeddings.LayerNorm.bias")
                           seq-len hidden-size layer-norm-eps)]
    (loop [x x, layer 0, blk (block-fn (array-dtype x))]
      (if (clojure.core/< layer num-layers)
        (let [p (str "encoder.layer." layer ".")]
          (recur (blk x
                   (w (str p "attention.self.query.weight"))
                   (w (str p "attention.self.query.bias"))
                   (w (str p "attention.self.key.weight"))
                   (w (str p "attention.self.key.bias"))
                   (w (str p "attention.self.value.weight"))
                   (w (str p "attention.self.value.bias"))
                   (w (str p "attention.output.dense.weight"))
                   (w (str p "attention.output.dense.bias"))
                   (w (str p "attention.output.LayerNorm.weight"))
                   (w (str p "attention.output.LayerNorm.bias"))
                   (w (str p "intermediate.dense.weight"))
                   (w (str p "intermediate.dense.bias"))
                   (w (str p "output.dense.weight"))
                   (w (str p "output.dense.bias"))
                   (w (str p "output.LayerNorm.weight"))
                   (w (str p "output.LayerNorm.bias"))
                   seq-len hidden-size num-heads)
                 (clojure.core/inc layer)
                 blk))
        x))))

(defn sentence-embedding
  "Sentence embedding via mean pooling + L2 normalization (the
  sentence-transformers default pooling). Returns float[hidden_size]."
  [model ^longs token-ids]
  (let [hidden (encode model token-ids)
        dim (long (:hidden-size model))
        seq-len (long (clojure.core/alength token-ids))]
    (nn/l2-normalize! (nn/mean-pool hidden seq-len dim) dim)))

;; ================================================================
;; Public API — load + embed (folded in from the former transformers.clj)
;; ================================================================

(defn from-pretrained
  "Load a BERT sentence-encoder from a local HF directory (model.safetensors,
  config.json, vocab.txt). Returns a model map usable by `embed`."
  [dir]
  (let [model (load-model dir)
        vocab (tok/load-vocab (str dir "/vocab.txt"))]
    (assoc model :vocab vocab :dir dir)))

(defn embed
  "Embed text → L2-normalized mean-pooled sentence embedding (float[hidden_size]).
  `text` is a string (one embedding) or a seq of strings (vector of embeddings).
  Tokenization is WordPiece with [CLS]/[SEP], lower-cased (BERT default)."
  [model text]
  (let [vocab (:vocab model)
        one (fn [s] (sentence-embedding model (tok/tokenize vocab s)))]
    (if (string? text) (one text) (mapv one text))))

(defn cosine-sim
  "Cosine similarity of two L2-normalized embeddings (= dot product)."
  [a b]
  (cosine-similarity a b (clojure.core/alength a)))
