(ns pretrained.arch.modernbert-gpu
  "Resident Level Zero execution for a fixed-shape ModernBERT encoder.

  Compilation is deliberately separate from model loading: a resident graph has
  a fixed sequence length, while Laya inputs are variable length.  The compiled
  value owns all layer weights on device and may be replayed for any unpadded
  sequence having that length.  Embedding lookup remains on the CPU for now;
  the much more expensive transformer stack and final norm are one composed GPU
  graph with no inter-layer host transfers."
  (:require [pretrained.arch.modernbert :as mb]
            [raster.dl.nn :as nn]
            [raster.gpu.compiled :as compiled]
            [raster.gpu.value :as value]))

(def ^:private first-layer-constants
  '[mlp-norm zero-bias wqkv wo wi w-mlp-out])

(def ^:private layer-constants
  '[attn-norm mlp-norm zero-bias wqkv wo wi w-mlp-out])

(defn- layer-args
  [model x segments seq-len layer]
  (let [{:keys [d-model d-ff n-heads head-dim global-every local-attention
                global-theta local-theta eps zero-bias]} model
        prefix (str "encoder.layers." layer ".")
        global? (zero? (mod layer global-every))
        tail [seq-len d-model d-ff n-heads head-dim
              (if global? global-theta local-theta)
              (/ 1.0 (Math/sqrt (double head-dim))) eps
              ;; A radius of seq-len is exactly full bidirectional attention.
              (if global? seq-len (inc (quot local-attention 2)))]]
    (if (zero? layer)
      (vec (concat
            (if segments [x segments] [x])
            [(mb/weight model (str prefix "mlp_norm.weight")) zero-bias
             (mb/weight model (str prefix "attn.Wqkv.weight"))
             (mb/weight model (str prefix "attn.Wo.weight"))
             (mb/weight model (str prefix "mlp.Wi.weight"))
             (mb/weight model (str prefix "mlp.Wo.weight"))]
            tail))
      (vec (concat
            (if segments [x segments] [x])
            [(mb/weight model (str prefix "attn_norm.weight"))
             (mb/weight model (str prefix "mlp_norm.weight")) zero-bias
             (mb/weight model (str prefix "attn.Wqkv.weight"))
             (mb/weight model (str prefix "attn.Wo.weight"))
             (mb/weight model (str prefix "mlp.Wi.weight"))
             (mb/weight model (str prefix "mlp.Wo.weight"))]
            tail)))))

(defn- lower-layer
  [model x segments seq-len layer opts]
  (compiled/lower
   (if segments
     (if (zero? layer)
       #'mb/modernbert-resident-segmented-first-block
       #'mb/modernbert-resident-segmented-block)
     (if (zero? layer)
       #'mb/modernbert-resident-first-block
       #'mb/modernbert-resident-block))
   (layer-args model x segments seq-len layer)
   (merge {:target :ze:0
           :dtype :float
           :constants (if (zero? layer) first-layer-constants layer-constants)
           :on-non-resident :throw}
          (select-keys opts [:target :gemm-precision :schedule]))))

(defn prepare-encoder
  "Lower and compose one fixed-`seq-len` encoder without contacting the GPU.

  The returned Raster Prepared artifact is useful for inspecting/caching the
  graph. `compile-encoder` instantiates it and uploads constants."
  ([model seq-len] (prepare-encoder model seq-len {}))
  ([model seq-len opts]
   (when-not (pos? (long seq-len))
     (throw (ex-info "ModernBERT GPU sequence length must be positive"
                     {:seq-len seq-len})))
   (let [{:keys [d-model n-layers eps zero-bias]} model
         _ (when-not (pos? (long n-layers))
             (throw (ex-info "ModernBERT GPU encoder must contain at least one layer"
                             {:n-layers n-layers})))
         x (float-array (* (long seq-len) d-model))
         segments (when (:segments? opts)
                    (or (:segments-placeholder opts) (float-array (long seq-len))))
         layers (mapv #(lower-layer model x segments (long seq-len) % opts)
                      (range n-layers))
         final-args [x (mb/weight model "encoder.final_norm.weight")
                     zero-bias (long seq-len) d-model eps]
         final (compiled/lower
                #'nn/layer-norm final-args
                (merge {:target :ze:0 :dtype :float
                        :constants '[gamma beta]
                        :on-non-resident :throw}
                       (select-keys opts [:target :schedule])))
         ids (mapv #(keyword (str "layer-" %)) (range n-layers))
         components (conj (mapv (fn [id program] {:id id :program program}) ids layers)
                          {:id :final-norm :program final})
         layer-out (mapv #(-> % :out-tree first :key) layers)
         final-out (-> final :out-tree first :key)
         connections
         (vec
          (concat
           (map (fn [i]
                  {:from [(nth ids i) (nth layer-out i)]
                   :to [(nth ids (inc i)) :x]})
                (range (dec n-layers)))
           [{:from [(peek ids) (peek layer-out)]
             :to [:final-norm :x]}]))
         zero-bias-share
         (conj (mapv (fn [id] [id :zero-bias]) ids)
               [:final-norm :beta])
         segment-share (when segments
                         (mapv (fn [id] [id :segments]) ids))]
     (compiled/compose
      {:id [:modernbert-encoder (or (:target opts) :ze:0) seq-len]
       :components components
       :connections connections
       :shares (cond-> [zero-bias-share] segment-share (conj segment-share))
       :outputs [{:key :encoded :from [:final-norm final-out]}]
       :attributes {:architecture :modernbert
                    :sequence-length seq-len
                    :layers n-layers
                    :segmented? (boolean segments)}}))))

(defn compile-encoder
  "Instantiate a fixed-shape resident encoder and upload its weights once."
  ([model seq-len] (compile-encoder model seq-len {}))
  ([model seq-len opts]
   (let [prepared (prepare-encoder model seq-len opts)]
     {:program (compiled/instantiate! prepared (select-keys opts [:profile?]))
      :prepared prepared
      :input-key [(keyword "layer-0") :x]
      :seq-len (long seq-len)
      :d-model (:d-model model)
      :model model})))

(defn encode-embeddings-device
  "Replay a compiled encoder for already normalized embeddings.
  Returns a resident DeviceArray, valid until the next replay or `close!`."
  [{:keys [program input-key seq-len d-model]} embeddings]
  (when-not (= (* seq-len d-model) (alength embeddings))
    (throw (ex-info "ModernBERT GPU embedding shape mismatch"
                    {:expected (* seq-len d-model)
                     :actual (alength embeddings)})))
  (:encoded (program {input-key embeddings})))

(defn encode-embeddings
  "Replay a compiled encoder and download its final hidden states."
  [encoder embeddings]
  (value/->host (encode-embeddings-device encoder embeddings)))

(defn encode
  "Embed one exact-length token sequence on CPU, execute all encoder layers on
  the GPU, and download final hidden states."
  [{:keys [model seq-len] :as encoder} token-ids]
  (when-not (= seq-len (count token-ids))
    (throw (ex-info "token sequence does not match compiled GPU shape"
                    {:compiled-length seq-len :actual-length (count token-ids)})))
  (let [{:keys [d-model vocab-size eps]} model
        ids (long-array token-ids)
        embedded (nn/embedding
                  (mb/weight model "encoder.embeddings.tok_embeddings.weight")
                  ids seq-len vocab-size d-model)
        normalized (nn/layer-norm
                    embedded (mb/weight model "encoder.embeddings.norm.weight")
                    (:zero-bias model) seq-len d-model eps)]
    (encode-embeddings encoder normalized)))

(defn close!
  "Release the resident graph, kernels, and device buffers."
  [{:keys [program]}]
  (compiled/close! program))
