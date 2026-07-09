(ns pretrained.arch.gemma3
  "Gemma 3 text architecture as a DATA descriptor over the generic decoder engine.
  The whole architecture is the map below — no per-arch forward code. Requiring this
  ns registers gemma3 with the loader."
  (:require [pretrained.decoder :as dec]
            [pretrained.loader :as loader]))

(def descriptor
  {:arch :gemma3
   :hf-arch #{"gemma3_text" "Gemma3ForCausalLM" "Gemma3ForConditionalGeneration" "gemma3"}
   ;; role -> HF tensor-name template (%d = layer index)
   :names {:embed          "embed_tokens.weight"
           :final-norm     "norm.weight"
           :attn-q         "layers.%d.self_attn.q_proj.weight"
           :attn-k         "layers.%d.self_attn.k_proj.weight"
           :attn-v         "layers.%d.self_attn.v_proj.weight"
           :attn-o         "layers.%d.self_attn.o_proj.weight"
           :attn-q-norm    "layers.%d.self_attn.q_norm.weight"
           :attn-k-norm    "layers.%d.self_attn.k_norm.weight"
           :attn-norm      "layers.%d.input_layernorm.weight"
           :attn-post-norm "layers.%d.post_attention_layernorm.weight"
           :ffn-pre-norm   "layers.%d.pre_feedforward_layernorm.weight"
           :ffn-post-norm  "layers.%d.post_feedforward_layernorm.weight"
           :ffn-gate       "layers.%d.mlp.gate_proj.weight"
           :ffn-up         "layers.%d.mlp.up_proj.weight"
           :ffn-down       "layers.%d.mlp.down_proj.weight"}
   :linear-roles        #{:attn-q :attn-k :attn-v :attn-o :ffn-gate :ffn-up :ffn-down}
   :global-linear-roles #{:embed}                 ;; tied lm-head, quantized once
   :flags {:norm {:type :rms :gain-offset 1.0}    ;; (1 + weight) rmsnorm
           :embed-scale :sqrt-d
           :tied-lm-head true
           :qk-norm true                           ;; per-head q/k rmsnorm
           :sandwich-norms true                    ;; post-attn + pre/post-ffn norms
           :ffn :geglu                             ;; gelu-gated
           :rope :dual                             ;; local (sliding) vs global per layer
           :global-layer-pattern 6                 ;; every 6th layer is full-attention
           :sliding-window {:size 512}}})

(defn build
  "Generic HF load + descriptor + quantize — no gemma-specific code."
  [dir _cfg]
  (let [m (assoc (dec/load-hf dir) :desc descriptor)]
    (assoc m :qm (dec/quantize-stream m))))

(loader/register-architecture!
 (:hf-arch descriptor)
 {:load build
  :generate (fn [model ids n sampler] (vec (dec/generate-cached model (long-array ids) n sampler)))})
