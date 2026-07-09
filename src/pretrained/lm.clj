(ns pretrained.lm
  "Unified decoder-LM text-generation API — the third of the pretrained.{embed,asr,lm}
  task trio, all shaped alike: (load-X :key [dir] [opts]) then a task verb.

    (require '[pretrained.lm :as lm])
    (def g (lm/load-lm :gemma-3-270m-it))                 ;; key -> HF auto-download (pretrained.hub)
    (lm/generate-text g \"The capital of France is\" 20)    ;; => \" Paris. ...\"
    (lm/load-lm :gemma-3-270m-it {:gpu? true})            ;; resident Intel-GPU decode (Level Zero/OpenCL)
    (lm/load-lm :gemma-3-270m-it \"/local/dir\")            ;; your own weights for a known model

  Curated registry only. To run an ARBITRARY HF checkpoint (any registered architecture
  family, no validation guarantee), drop to the low-level pretrained.loader/from-pretrained
  on the model dir directly — that is the advanced, unvalidated escape hatch."
  (:require [pretrained.loader :as loader]))

(def registry
  "Decoder-LM keys -> HF repo + architecture ns. gemma-3 is anchor-validated (token-exact
  vs oracle); the others share the same descriptor-driven decode engine and add a checkpoint."
  {:gemma-3-270m-it       {:hf "google/gemma-3-270m-it" :arch 'pretrained.arch.gemma3}
   :gemma-3-1b-it         {:hf "google/gemma-3-1b-it"   :arch 'pretrained.arch.gemma3}
   :qwen3-0.6b            {:hf "Qwen/Qwen3-0.6B"        :arch 'pretrained.arch.qwen3}
   :qwen3-1.7b            {:hf "Qwen/Qwen3-1.7B"        :arch 'pretrained.arch.qwen3}
   :smollm2-360m-instruct {:hf "HuggingFaceTB/SmolLM2-360M-Instruct" :arch 'pretrained.arch.llama}
   :smollm2-135m-instruct {:hf "HuggingFaceTB/SmolLM2-135M-Instruct" :arch 'pretrained.arch.llama}})

(defn registered "Sorted decoder-LM registry keys." [] (sort (keys registry)))

(defn load-lm
  "Load a decoder LM by registry key. Weights auto-download from HF into the local cache
  on first use (pretrained.hub); pass an explicit local dir to skip the download (your own
  weights for a known model). opts: {:gpu? true} for resident Intel-GPU decode,
  :maxpos KV-cache length (GPU, default 2048)."
  ([k] (load-lm k nil {}))
  ([k dir-or-opts]
   (if (map? dir-or-opts) (load-lm k nil dir-or-opts) (load-lm k dir-or-opts {})))
  ([k dir opts]
   (let [entry (get registry k)
         _ (assert entry (str "unknown LM " k " — known: " (registered)))
         _ (when-let [a (:arch entry)] (require a))
         dir (or dir ((requiring-resolve 'pretrained.hub/ensure-model) (:hf entry)))
         base (assoc (loader/from-pretrained dir) ::entry entry)]
     (if (:gpu? opts)
       (do (require 'pretrained.decoder-gpu)
           (assoc base ::mode :gpu
                  ::decode ((requiring-resolve 'pretrained.decoder-gpu/bind-decode!)
                            base :maxpos (long (or (:maxpos opts) 2048)))))
       (assoc base ::mode :cpu)))))

(defn generate-ids
  "Prompt token-ids -> n generated ids. CPU honors the sampler `opts`
  ({:temperature :top-k :top-p :seed}; empty = greedy); GPU decode is greedy."
  ([m prompt-ids n] (generate-ids m prompt-ids n {}))
  ([m prompt-ids n opts]
   (if (= :gpu (::mode m))
     ((requiring-resolve 'pretrained.decoder-gpu/generate-resident) (::decode m) (vec prompt-ids) n)
     (loader/generate-ids m prompt-ids n opts))))

(defn generate-text
  "Encode `prompt`, generate n tokens, decode to a string. See generate-ids for opts."
  ([m prompt n] (generate-text m prompt n {}))
  ([m prompt n opts]
   (if (= :gpu (::mode m))
     (let [{:keys [tok encode decode]} (:tokenizer m)
           ids (vec (encode tok prompt))
           out (generate-ids m ids n opts)]
       (decode tok out))
     (loader/generate-text m prompt n opts))))
