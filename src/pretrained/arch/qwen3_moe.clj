(ns pretrained.arch.qwen3-moe
  "Qwen3-MoE (Qwen3MoeForCausalLM) — Qwen3-Coder-30B-A3B-Instruct and the
  Qwen3-30B-A3B-Instruct/Thinking-2507 twins (field-identical configs).

  Attention/norms/rope = EXACTLY the qwen3 stack this engine already runs.
  The delta is the FFN: per layer, a router linear [experts, d] followed by
  softmax over ALL experts -> top-k of the probabilities -> renormalize
  (norm_topk_prob) -> weighted sum of k expert SwiGLU FFNs (moe_intermediate
  width). No shared expert; every layer is MoE (decoder_sparse_step 1).

  Weights: router (mlp.gate) + norms + embed stay f32; expert FFNs + attention
  projections quantize to the stream format (Q4 for the 30B, ~17GB)."
  (:require [pretrained.decoder :as dec]
            [pretrained.loader :as loader]
            [pretrained.hub :as hub]
            [pretrained.safetensors :as st]
            [raster.compiler.backend.cpu.quant :as cq]
            [clojure.java.io :as io]
            [clojure.string :as string]
            [clojure.data.json :as json]
            [pretrained.arch.qwen3 :as qwen3]))

(def descriptor
  (-> qwen3/descriptor
      (assoc :arch :qwen3-moe
             :hf-arch #{"qwen3_moe" "Qwen3MoeForCausalLM"})
      (assoc-in [:names :moe-router] "layers.%d.mlp.gate.weight")
      (update :names dissoc :ffn-gate :ffn-up :ffn-down)
      (assoc :moe-names {:gate "layers.%d.mlp.experts.%d.gate_proj.weight"
                         :up   "layers.%d.mlp.experts.%d.up_proj.weight"
                         :down "layers.%d.mlp.experts.%d.down_proj.weight"})
      (assoc :linear-roles #{:attn-q :attn-k :attn-v :attn-o})
      (assoc :global-linear-roles #{:lm-head})
      (assoc-in [:flags :tied-lm-head] false)
      (assoc-in [:flags :moe] true)))


;; ---------------------------------------------------------------------------
;; Shard-streaming fetch+quantize: the 61GB bf16 checkpoint never exists on disk
;; whole. Per shard: download (~4GB) -> quantize its linears into the stream
;; file / keep small tensors f32 in keepers.bin -> DELETE the shard.
;; End state: ~12GB Q4 streams + ~1.4GB f32 keepers.
;; ---------------------------------------------------------------------------

(defn- write-arr-tagged! [^java.io.DataOutputStream out a]
  (cond
    (bytes? a) (do (.writeByte out 0) (.writeInt out (alength ^bytes a)) (.write out ^bytes a))
    (instance? (Class/forName "[I") a)
    (let [^ints a a bb (java.nio.ByteBuffer/allocate (* 4 (alength a)))]
      (.writeByte out 1) (.writeInt out (alength a)) (.put (.asIntBuffer bb) a) (.write out (.array bb)))
    (instance? (Class/forName "[F") a)
    (let [^floats a a bb (java.nio.ByteBuffer/allocate (* 4 (alength a)))]
      (.writeByte out 2) (.writeInt out (alength a)) (.put (.asFloatBuffer bb) a) (.write out (.array bb)))))

(defn- quantizable? [nm]
  (or (re-matches #"model\.layers\.\d+\.self_attn\.[qkvo]_proj\.weight" nm)
      (re-matches #"model\.layers\.\d+\.mlp\.experts\.\d+\.(gate|up|down)_proj\.weight" nm)
      (= nm "lm_head.weight")))

(defn fetch-and-quantize!
  "Stream the sharded checkpoint through quantization. Writes into `dir`:
  .qstream-<fmt>.bin (quantized linears, decoder stream format), keepers.bin
  (f32 embed/norms/router), plus config/tokenizer files. Resumable per shard."
  [repo dir & {:keys [fmt] :or {fmt :q4}}]
  (let [dir (io/file dir)
        _ (.mkdirs dir)
        info (hub/get-json (str "https://huggingface.co/api/models/" repo))
        sha (get info "sha")
        tree (hub/get-json (str "https://huggingface.co/api/models/" repo "/tree/main?recursive=true"))
        by-path (into {} (map (fn [f] [(get f "path") f]) tree))
        small ["config.json" "generation_config.json" "tokenizer.json" "tokenizer_config.json"
               "chat_template.jinja" "merges.txt" "vocab.json" "model.safetensors.index.json"]
        _ (doseq [f small :when (by-path f)]
            (hub/download-file! repo sha (by-path f) (str dir)))
        index (json/read-str (slurp (io/file dir "model.safetensors.index.json")))
        shards (sort (distinct (vals (get index "weight_map"))))
        format* (case fmt :q4 cq/q4-0 :q8 cq/q8-0)
        qf (io/file dir (str ".qstream-" (name fmt) ".bin"))
        kf (io/file dir "keepers.bin")
        done-f (io/file dir ".shards-done")
        done (if (.exists done-f) (set (string/split-lines (slurp done-f))) #{})
        qout (java.io.DataOutputStream. (java.io.BufferedOutputStream.
                                         (java.io.FileOutputStream. qf (boolean (seq done))) (* 1024 1024)))
        kout (java.io.DataOutputStream. (java.io.BufferedOutputStream.
                                         (java.io.FileOutputStream. kf (boolean (seq done))) (* 1024 1024)))
        nq (atom 0)]
    (try
      (doseq [shard shards :when (not (done shard))]
        (println (str "[shard] " shard))
        (hub/download-file! repo sha (by-path shard) (str dir))
        (let [lz (st/load-safetensors-lazy (str dir "/" shard))]
          (doseq [[nm _] (sort-by key (:tensors lz))]
            (let [{:keys [^floats data shape]} (st/read-tensor lz nm)]
              (if (quantizable? nm)
                (let [[out in] shape
                      bare (if (.startsWith ^String nm "model.") (subs nm 6) nm)
                      {:keys [wq ws]} (cq/quantize-weight data format*)
                      e (cq/repack-stream wq ws (long out) (long in) format*)]
                  (.writeUTF qout bare)
                  (.writeInt qout (int in)) (.writeInt qout (int out))
                  (write-arr-tagged! qout (:wqi e))
                  (write-arr-tagged! qout (:wsi e))
                  (swap! nq inc))
                (let [bare (if (.startsWith ^String nm "model.") (subs nm 6) nm)]
                  (.writeUTF kout bare)
                  (.writeInt kout (count shape))
                  (doseq [d shape] (.writeInt kout (int d)))
                  (write-arr-tagged! kout data))))))
        (.flush qout) (.flush kout)
        (io/delete-file (io/file dir shard))
        (spit done-f (str shard "\n") :append true))
      (finally (.close qout) (.close kout)))
    (println "DONE:" @nq "quantized tensors;"
             (format "%.1fGB streams, %.1fGB keepers"
                     (/ (.length qf) 1e9) (/ (.length kf) 1e9)))))


(defn- read-arr-tagged [^java.io.DataInputStream in]
  (let [tag (.readByte in) n (.readInt in)]
    (case (int tag)
      0 (let [b (byte-array n)] (.readFully in b) b)
      1 (let [b (byte-array (* 4 n)) o (int-array n)] (.readFully in b)
          (.get (.asIntBuffer (java.nio.ByteBuffer/wrap b)) o) o)
      2 (let [b (byte-array (* 4 n)) o (float-array n)] (.readFully in b)
          (.get (.asFloatBuffer (java.nio.ByteBuffer/wrap b)) o) o))))

(defn- read-stream-entries
  "Headerless entry stream written by fetch-and-quantize! → {name {:wqi :wsi :in :out}}."
  [f]
  (with-open [in (java.io.DataInputStream. (java.io.BufferedInputStream.
                                            (java.io.FileInputStream. ^java.io.File (clojure.java.io/file f)) (* 4 1024 1024)))]
    (loop [acc (transient {})]
      (let [nm (try (.readUTF in) (catch java.io.EOFException _ nil))]
        (if nm
          (let [i (.readInt in) o (.readInt in)
                wqi (read-arr-tagged in) wsi (read-arr-tagged in)]
            (recur (assoc! acc nm {:wqi wqi :wsi wsi :in (long i) :out (long o)})))
          (persistent! acc))))))

(defn- read-keepers
  "keepers.bin → {name {:shape [...] :data float[]}}."
  [f]
  (with-open [in (java.io.DataInputStream. (java.io.BufferedInputStream.
                                            (java.io.FileInputStream. ^java.io.File (clojure.java.io/file f)) (* 4 1024 1024)))]
    (loop [acc (transient {})]
      (let [nm (try (.readUTF in) (catch java.io.EOFException _ nil))]
        (if nm
          (let [rank (.readInt in)
                shape (vec (repeatedly rank #(.readInt in)))
                data (read-arr-tagged in)]
            (recur (assoc! acc nm {:shape shape :data data})))
          (persistent! acc))))))

(defn load-quantized
  "Load a fetch-and-quantize!'d model dir: f32 keepers (embed/norms/router) +
  quantized streams. Never touches the original checkpoint."
  [dir & {:keys [fmt] :or {fmt :q4}}]
  (let [cfg (json/read-str (slurp (str dir "/config.json")) :key-fn keyword)
        head-dim (or (:head_dim cfg) (quot (:hidden_size cfg) (:num_attention_heads cfg)))
        moe {:experts (long (:num_experts cfg)) :top-k (long (:num_experts_per_tok cfg))
             :inter (long (:moe_intermediate_size cfg)) :norm-topk? (boolean (:norm_topk_prob cfg))}
        m {:config cfg :dir dir
           :weights (read-keepers (str dir "/keepers.bin"))
           :desc (assoc-in descriptor [:flags :moe] moe)
           :n-layers (:num_hidden_layers cfg) :d-model (:hidden_size cfg)
           :d-ff (:intermediate_size cfg) :n-q (:num_attention_heads cfg)
           :n-kv (:num_key_value_heads cfg) :head-dim head-dim
           :vocab (:vocab_size cfg) :eps (double (:rms_norm_eps cfg))
           :rope-global (double (or (get-in cfg [:rope_parameters :rope_theta]) (:rope_theta cfg) 1.0e7))
           :rope-local 10000.0
           :attn-scale (/ 1.0 (Math/sqrt (double head-dim)))
           :tokenizer (loader/detect-tokenizer dir)}]
    (assoc m :qm {:fmt fmt :w (read-stream-entries (str dir "/.qstream-" (name fmt) ".bin"))})))

(defn build [dir cfg]
  (let [m (dec/load-hf dir)
        moe {:experts (long (:num_experts (:config m)))
             :top-k (long (:num_experts_per_tok (:config m)))
             :inter (long (:moe_intermediate_size (:config m)))
             :norm-topk? (boolean (:norm_topk_prob (:config m)))}
        m (assoc m :desc (assoc-in descriptor [:flags :moe] moe))]
    (assoc m :qm (dec/quantize-stream m (or (:quant cfg) :q4)))))

(loader/register-architecture!
 (:hf-arch descriptor)
 {:load build
  :generate (fn [m prompt-ids n opts]
              (dec/generate-cached m prompt-ids n (or (:sampler opts)
                                                      (requiring-resolve 'pretrained.sampling/greedy))))})
