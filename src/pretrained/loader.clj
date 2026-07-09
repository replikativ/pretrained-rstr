(ns pretrained.loader
  "Generic HuggingFace model loader: read config.json, dispatch on model_type to a
  registered architecture, auto-detect the tokenizer, and return a runnable model
  with a text->text generate. The path toward 'load and run arbitrary HF models' —
  each new architecture is one register-architecture! entry reusing the raster.dl
  building blocks; each new tokenizer family one more branch in detect-tokenizer.

  This is the LOW-LEVEL, generic loader — it dispatches ANY model dir on config.json
  with no validation guarantee. For curated, validated models with HF auto-download,
  use the task-level pretrained.lm/load-lm (which wraps this). Bundled architecture
  namespaces are auto-required on a registry miss, so no manual (require ...) is needed.

  Usage (advanced / bring-your-own-checkpoint):
    (def m (from-pretrained \"/path/to/gemma-3-270m-it\"))
    (generate-text m \"The capital of France is\" 20)"
  (:require [clojure.data.json :as json]
            [pretrained.sampling :as samp]
            [pretrained.tokenizer.sp :as sp]
            [pretrained.tokenizer.bpe :as bpe]))

;; ---------------------------------------------------------------------------
;; Architecture registry
;; ---------------------------------------------------------------------------

(defonce ^:private arch-registry (atom {}))

(defn register-architecture!
  "Register an architecture. `model-types` is a set of HF identifiers matched against
  config.json's model_type AND architectures[]. `handler` is a map:
    :load     (fn [dir cfg] -> model-map)   ;; loads weights, prepares for decode
    :generate (fn [model token-ids n] -> new-token-ids)"
  [model-types handler]
  (swap! arch-registry into (zipmap model-types (repeat handler)))
  (keys @arch-registry))

(defn registered-architectures [] (sort (keys @arch-registry)))

(defn- dispatch-key
  "Pick the registry key for a config: try model_type then each architecture name."
  [cfg]
  ;; architectures first: they are MORE SPECIFIC than model_type (Gemma3TextModel =
  ;; EmbeddingGemma encoder vs Gemma3ForCausalLM both carry model_type gemma3_text).
  (let [cands (concat (:architectures cfg) [(:model_type cfg)])
        reg @arch-registry]
    (some #(when (get reg %) %) (filter some? cands))))

;; ---------------------------------------------------------------------------
;; Tokenizer auto-detection
;; ---------------------------------------------------------------------------

(defn detect-tokenizer
  "Choose a tokenizer for the model dir. SentencePiece-style (byte_fallback) BPE for
  Gemma/Llama/Mistral; GPT-2 byte-level BPE otherwise. Returns {:kind :encode :decode}."
  [dir]
  (let [tj (str dir "/tokenizer.json")]
    (when (.exists (java.io.File. tj))
      (let [model (get (json/read-str (slurp tj)) "model")]
        (if (get model "byte_fallback")
          {:kind :sp :tok (sp/load-tokenizer tj)
           :encode sp/encode :decode sp/decode}
          {:kind :bpe :tok (bpe/load-bpe-tokenizer tj)
           :encode bpe/encode :decode bpe/decode})))))

;; ---------------------------------------------------------------------------
;; Public API
;; ---------------------------------------------------------------------------

(def ^:private bundled-arch-nses
  "Architecture namespaces whose load registers a handler via register-architecture!.
  Required lazily on a registry miss so callers don't have to know to `(require ...)`
  the right arch ns first (load-embedder already auto-requires; from-pretrained now
  matches that ergonomics). require is cached, so this is a one-time cost."
  '[pretrained.arch.gemma3 pretrained.arch.embedding-gemma pretrained.arch.llama
    pretrained.arch.qwen3 pretrained.arch.qwen3-moe])

(defn- ensure-bundled-archs! []
  (doseq [ns bundled-arch-nses] (try (require ns) (catch Throwable _))))

(defn from-pretrained
  "Load an HF model directory. Dispatches on config model_type/architectures to a
  registered architecture handler, loads the model + tokenizer. Returns a model map
  with :arch, :handler and :tokenizer attached. Bundled architecture namespaces are
  auto-required on a registry miss, so callers need not require the arch ns first."
  [dir]
  (let [cfg (json/read-str (slurp (str dir "/config.json")) :key-fn keyword)
        k (or (dispatch-key cfg)
              (do (ensure-bundled-archs!) (dispatch-key cfg))
              (throw (ex-info (str "No architecture registered for "
                                   (pr-str [(:model_type cfg) (:architectures cfg)])
                                   ". Registered: " (registered-architectures))
                              {:config cfg})))
        handler (get @arch-registry k)
        model ((:load handler) dir cfg)]
    (assoc model :arch k :handler handler :tokenizer (detect-tokenizer dir))))

(defn generate-ids
  "Generate n new token ids from a prompt id seq. `opts` is a sampler config
  {:temperature :top-k :top-p :seed} (empty/absent = greedy)."
  ([model prompt-ids n] (generate-ids model prompt-ids n {}))
  ([model prompt-ids n opts]
   ((:generate (:handler model)) model prompt-ids n (samp/make-sampler opts))))

(defn generate-text
  "Encode `prompt`, generate `n` tokens (sampler from `opts`), decode to a string."
  ([model prompt n] (generate-text model prompt n {}))
  ([model prompt n opts]
   (let [{:keys [tok encode decode]} (:tokenizer model)
         _ (when-not tok (throw (ex-info "model has no tokenizer" {:arch (:arch model)})))
         ids (encode tok prompt)
         new-ids (generate-ids model ids n opts)]
     (decode tok new-ids))))
