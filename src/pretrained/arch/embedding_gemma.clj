(ns pretrained.arch.embedding-gemma
  "EmbeddingGemma-300M (google/embeddinggemma-300m; ungated mirror unsloth/…) as a
  DATA descriptor. The gemma3 layer stack unchanged (RMSNorm (1+w), qk-norm,
  sandwich norms, GeGLU, dual rope 5:1 + sliding 512, sqrt-d embed scaling, same
  262144 SentencePiece vocab) with ONE backbone change — BIDIRECTIONAL attention
  (Gemma3TextModel encoder, `use_bidirectional_attention`) — plus the
  sentence-transformers head: mean pooling (prompt included) → Dense 768→3072 →
  Dense 3072→768 (no bias, identity) → L2 normalize. Matryoshka: truncate to
  512/256/128 then re-normalize.

  Dims: 24L, d=768, 3Q/1KV heads, head-dim 256, d-ff 1152, max seq 2048.
  Runs on the GPU prefill mode (bind-embed!/embed-gpu, T ≤ 512 so the symmetric
  sliding window never binds — exact). Input: <bos> prompt+text <eos> with the
  task prompts, e.g. query = \"task: search result | query: \" and document =
  \"title: none | text: \" (trailing spaces significant)."
  (:require [pretrained.decoder :as dec]
            [pretrained.decoder-gpu :as dg]
            [pretrained.loader :as loader]))

(def descriptor
  {:arch :embedding-gemma
   :hf-arch #{"Gemma3TextModel"}   ;; the encoder export; causal gemma3 keeps gemma3_text
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
   :global-linear-roles #{}                      ;; no lm-head — encoder only
   :flags {:norm {:type :rms :gain-offset 1.0}
           :embed-scale :sqrt-d
           :qk-norm true
           :sandwich-norms true
           :ffn :geglu
           :rope :dual
           :global-layer-pattern 6
           :sliding-window {:size 512}
           :bidirectional? true
           :pooling :mean}})

(def prompts
  "sentence-transformers task prompts (trailing spaces significant)."
  {:query    "task: search result | query: "
   :document "title: none | text: "})

(defn build [dir _cfg]
  (-> (dec/load-hf dir)
      (assoc :desc descriptor
             :dense (dg/load-dense-head dir))))

(loader/register-architecture!
 (:hf-arch descriptor)
 {:load build
  :generate (fn [_ _ _ _] (throw (ex-info "EmbeddingGemma is an encoder — use embed, not generate" {})))})
