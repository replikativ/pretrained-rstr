(ns pretrained.arch.llama
  "Llama / SmolLM / TinyLlama family as a DATA descriptor over the generic decoder
  engine — the second architecture, expressed entirely as the differences from
  gemma3 (all flags), with zero new forward code. This is the generality test:
  a standard decoder-LM should be addable as data alone.

  Differences from gemma3 (all captured below):
    - plain RMSNorm (no (1+w) gain offset)        :gain-offset 0.0
    - no QK-norm                                  :qk-norm false
    - only pre-attn + pre-ffn norms (no sandwich) :sandwich-norms false
    - SwiGLU (silu-gated) FFN                      :ffn :swiglu
    - single RoPE base, no sliding window         :rope :single
    - no embedding scaling                        :embed-scale nil
  Note: Llama names the pre-FFN norm post_attention_layernorm — handled by the
  :ffn-pre-norm role pointing there. Untied lm_head uses the :lm-head role.

  NOTE: validated end-to-end requires a llama-arch checkpoint locally (e.g.
  HuggingFaceTB/SmolLM2-360M-Instruct); the descriptor is the architecture."
  (:require [pretrained.decoder :as dec]
            [pretrained.loader :as loader]))

(def descriptor
  {:arch :llama
   :hf-arch #{"llama" "LlamaForCausalLM"}
   :names {:embed          "embed_tokens.weight"
           :lm-head        "lm_head.weight"      ;; used when not tied
           :final-norm     "norm.weight"
           :attn-q         "layers.%d.self_attn.q_proj.weight"
           :attn-k         "layers.%d.self_attn.k_proj.weight"
           :attn-v         "layers.%d.self_attn.v_proj.weight"
           :attn-o         "layers.%d.self_attn.o_proj.weight"
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
           :qk-norm false
           :sandwich-norms false
           :ffn :swiglu
           :rope :single}})

(defn build [dir _cfg]
  (let [m (assoc (dec/load-hf dir) :desc descriptor)]
    (assoc m :qm (dec/quantize-stream m))))

(loader/register-architecture!
 (:hf-arch descriptor)
 {:load build
  :generate (fn [model ids n sampler] (vec (dec/generate-cached model (long-array ids) n sampler)))})
