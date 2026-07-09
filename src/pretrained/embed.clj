(ns pretrained.embed
  "Text-embedding API over the descriptor engine.

  (embed-texts model texts) → {:data float[n*dim] :n :dim} — a packed row-major f32
  matrix of L2-normalized embeddings. Two adapters cover the downstream contracts:
    rows          → vector of per-row float[]  (proximum insert-batch; use :cosine)
    flat-doubles  → one flat row-major double[] (umap.rstr/fit X n dim)

  Decoder-embedders (Qwen3-Embedding) pool the LAST token's hidden state after final
  norm (dec/embed-prefill); the EOS token is appended per the Qwen3-Embedding
  convention, and queries may carry an instruction prefix (:instruct). Weights load
  in Q8_0 (measured: 4-bit weight quant costs ~5% embedding cosine, Q8_0 is
  lossless; see the qwen3 arch ns).

  BERT-family encoders (MiniLM/bge/mxbai — mean-pool) are the encoder counterpart
  to the decoder engine; :engine :encoder registry entries load via
  pretrained.arch.bert (self-contained: raster.dl blocks + safetensors + WordPiece)."
  (:require [pretrained.loader :as loader]
            [pretrained.decoder :as dec]))

;; ---------------------------------------------------------------------------
;; Model registry — name → how to load + pooling convention
;; ---------------------------------------------------------------------------

(def registry
  {:qwen3-embedding-0.6b
   {:hf "Qwen/Qwen3-Embedding-0.6B" :arch 'pretrained.arch.qwen3
    :engine :decoder :pooling :last-token :eos-append? true :dim 1024}
   :qwen3-embedding-0.6b-gpu
   {:hf "Qwen/Qwen3-Embedding-0.6B" :arch 'pretrained.arch.qwen3
    :engine :decoder-gpu :pooling :last-token :eos-append? true :dim 1024}
   :embeddinggemma-300m
   {:hf "unsloth/embeddinggemma-300m" :arch 'pretrained.arch.embedding-gemma
    :engine :decoder-gpu :pooling :mean :eos-append? true :dim 768
    ;; sentence-transformers task prompts (trailing spaces significant)
    :prompts {:query "task: search result | query: " :document "title: none | text: "}}
   :all-minilm-l6-v2
   {:hf "sentence-transformers/all-MiniLM-L6-v2"
    :engine :encoder :pooling :mean :dim 384}
   :bge-small-en-v1.5
   {:hf "BAAI/bge-small-en-v1.5"
    :engine :encoder :pooling :mean :dim 384}})

(defn load-embedder
  "Load an embedding model by registry key — weights auto-download from HF into
  the local cache on first use (pretrained.hub). Pass an explicit dir to skip."
  ([k]
   (let [entry (get registry k)]
     (assert entry (str "unknown embedder " k " — known: " (keys registry)))
     (load-embedder k ((requiring-resolve 'pretrained.hub/ensure-model) (:hf entry)))))
  ([k dir]
   (let [entry (get registry k)]
     (assert entry (str "unknown embedder " k " — known: " (keys registry)))
     (case (:engine entry)
       :decoder (do (require (:arch entry))
                    (assoc (loader/from-pretrained dir) ::entry entry))
       ;; GPU prefill engine: quantize (signed q8) + bind the resident T-block program.
       ;; opts: :T block size (max tokens per text incl. prompt; default 128).
       :decoder-gpu
       (let [_ (require (:arch entry) 'pretrained.decoder-gpu)
             bind-embed (requiring-resolve 'pretrained.decoder-gpu/bind-embed!)
             quantize (requiring-resolve 'pretrained.decoder-gpu/quantize-q8s)
             m (assoc (loader/from-pretrained dir) ::entry entry)]
         (assoc m ::gpu (bind-embed m :T (or (:T entry) 128) :qw (quantize m))))
       :encoder
       (do (require 'pretrained.arch.bert)
           {::entry entry
            ::encoder-model ((requiring-resolve 'pretrained.arch.bert/from-pretrained) dir)})))))

;; ---------------------------------------------------------------------------
;; Embedding
;; ---------------------------------------------------------------------------

(defn- encode-ids
  "Token ids for one text under the model's embedding convention. `kind` selects a
  registry prompt prefix (:query/:document — EmbeddingGemma style); `instruct` is the
  Qwen3-style instruction prefix. Both are optional."
  [model text instruct kind]
  (let [{:keys [tok encode]} (:tokenizer model)
        entry (::entry model)
        prefix (get (:prompts entry) (or kind :document) "")
        s (cond instruct (str "Instruct: " instruct "\nQuery: " text)
                (seq prefix) (str prefix text)
                :else text)
        ids (vec (encode tok s))]
    (if (:eos-append? entry)
      (conj ids (long (or (:eos_token_id (:config model)) 0)))
      ids)))

(defn- embed-one ^floats [model text instruct kind]
  (cond
    (::encoder-model model)
    (let [e ((requiring-resolve 'pretrained.arch.bert/embed) (::encoder-model model) text)]
      (if (instance? (Class/forName "[F") e) e (float-array e)))
    (::gpu model)
    ((requiring-resolve 'pretrained.decoder-gpu/embed-gpu)
     (::gpu model) (encode-ids model text instruct kind))
    :else
    (dec/embed-prefill model (encode-ids model text instruct kind))))

(defn embed-texts
  "Embed a string or seq of strings → {:data float[n*dim] :n :dim} (packed row-major,
  L2-normalized rows). :instruct adds the query-side instruction prefix (decoder
  embedders; leave nil for documents)."
  [model texts & {:keys [instruct kind]}]
  (let [texts (if (string? texts) [texts] (vec texts))
        n (count texts)
        ^floats e0 (embed-one model (first texts) instruct kind)
        dim (alength e0)
        out (float-array (* n dim))]
    (System/arraycopy e0 0 out 0 dim)
    (dotimes [i (dec n)]
      (System/arraycopy ^floats (embed-one model (nth texts (inc i)) instruct kind)
                        0 out (* (inc i) dim) dim))
    {:data out :n n :dim dim}))

;; ---------------------------------------------------------------------------
;; Adapters — the two downstream contracts
;; ---------------------------------------------------------------------------

(defn rows
  "Per-row float[] vectors — proximum's Vector type (create the index with
  :distance :cosine; rows are already L2-normalized)."
  [{:keys [^floats data n dim]}]
  (mapv (fn [i] (java.util.Arrays/copyOfRange data (int (* (long i) dim))
                                              (int (* (inc (long i)) dim))))
        (range n)))

(defn flat-doubles
  "One flat row-major double[n*dim] — umap.rstr/fit's X input (pass :n and :dim)."
  ^doubles [{:keys [^floats data]}]
  (let [n (alength data) out (double-array n)]
    (dotimes [i n] (aset out i (double (aget data i))))
    out))
