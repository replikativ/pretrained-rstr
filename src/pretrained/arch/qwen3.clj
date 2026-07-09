(ns pretrained.arch.qwen3
  "Qwen3 family (incl. Qwen3-Embedding) as a DATA descriptor over the generic decoder
  engine — no new forward code. Structurally a SUBSET of gemma3's flag space plus
  llama's plainness:
    - plain RMSNorm (no (1+w) offset)          :gain-offset 0.0
    - per-head QK-norm (like gemma3)           :qk-norm true
    - only pre-attn + pre-ffn norms            :sandwich-norms false
    - SwiGLU (silu-gated) FFN                  :ffn :swiglu
    - single RoPE base (theta from config,     :rope :single
      1e6 for Qwen3), no sliding window
    - no embedding scaling                     :embed-scale nil
    - tied lm-head (Qwen3-Embedding never
      uses it — last-token pooling instead)    :tied-lm-head true

  Qwen3-Embedding-0.6B: 28L, d=1024, 16Q/8KV heads, head-dim 128, d-ff 3072,
  vocab 151669. Embeddings = dec/embed-prefill (last-token pooling + L2 norm)."
  (:require [pretrained.decoder :as dec]
            [pretrained.loader :as loader]))

(def descriptor
  {:arch :qwen3
   :hf-arch #{"qwen3" "Qwen3ForCausalLM"}
   :names {:embed          "embed_tokens.weight"
           :lm-head        "lm_head.weight"      ;; used when not tied
           :final-norm     "norm.weight"
           :attn-q         "layers.%d.self_attn.q_proj.weight"
           :attn-k         "layers.%d.self_attn.k_proj.weight"
           :attn-v         "layers.%d.self_attn.v_proj.weight"
           :attn-o         "layers.%d.self_attn.o_proj.weight"
           :attn-q-norm    "layers.%d.self_attn.q_norm.weight"
           :attn-k-norm    "layers.%d.self_attn.k_norm.weight"
           :attn-norm      "layers.%d.input_layernorm.weight"
           :ffn-pre-norm   "layers.%d.post_attention_layernorm.weight"
           :ffn-gate       "layers.%d.mlp.gate_proj.weight"
           :ffn-up         "layers.%d.mlp.up_proj.weight"
           :ffn-down       "layers.%d.mlp.down_proj.weight"}
   :linear-roles        #{:attn-q :attn-k :attn-v :attn-o :ffn-gate :ffn-up :ffn-down}
   :global-linear-roles #{:embed}
   :flags {:norm {:type :rms :gain-offset 0.0}
           :embed-scale nil
           :tied-lm-head true
           :qk-norm true
           :sandwich-norms false
           :ffn :swiglu
           :rope :single}})

(defn build [dir _cfg]
  (let [m (assoc (dec/load-hf dir) :desc descriptor)]
    ;; :q8 — embedding quality needs 8-bit weights (Q4 costs ~5% cosine, measured)
    (assoc m :qm (dec/quantize-stream m :q8))))

(loader/register-architecture!
 (:hf-arch descriptor)
 {:load build
  :generate (fn [model ids n sampler] (vec (dec/generate-cached model (long-array ids) n sampler)))})
