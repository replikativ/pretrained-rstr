(ns pretrained.anchors-test
  "Model-gated integration anchors — each skips cleanly when its weights are not
  on disk (set PRETRAINED_MODELS or use ~/Development/models). These reproduce
  the port validations: Moonshine jfk transcript (character-exact vs HF torch),
  Qwen3-ASR jfk transcript, Qwen3-Embedding retrieval structure.

  Heavy: run individually, not in default CI. Each loads a full model."
  (:require [clojure.test :refer [deftest is testing]]
            [clojure.string :as str]))

(defn- gpu-up? []
  (try ((requiring-resolve 'raster.gpu.ze-runtime/init!)) true (catch Throwable _ false)))

(def ^:private models-dir
  (or (System/getenv "PRETRAINED_MODELS")
      (str (System/getProperty "user.home") "/Development/models")))

(defn- mdir [name*] (str models-dir "/" name*))
(defn- have? [name*] (.exists (java.io.File. (mdir name*))))
(def ^:private jfk (mdir "audio-samples/jfk.wav"))
(def ^:private jfk-text
  "And so my fellow Americans, ask not what your country can do for you, ask what you can do for your country.")

(deftest ^:anchors moonshine-jfk-anchor
  (if-not (and (have? "moonshine-streaming-medium") (.exists (java.io.File. jfk)))
    (println "SKIP moonshine anchor (weights not present)")
    (let [ms (requiring-resolve 'pretrained.asr.moonshine/load-model)
          tr (requiring-resolve 'pretrained.asr.moonshine/transcribe)
          m (ms (mdir "moonshine-streaming-medium"))]
      (testing "character-exact vs HF torch reference"
        (is (= jfk-text (tr m jfk))))
      (testing "streaming final == batch"
        (let [si (requiring-resolve 'pretrained.asr.moonshine/stream-init)
              sp (requiring-resolve 'pretrained.asr.moonshine/stream-push!)
              audio ((requiring-resolve 'pretrained.audio/load-wav) jfk)
              chunks (mapv (fn [i] (java.util.Arrays/copyOfRange
                                    ^floats (:samples audio) (* i 16000)
                                    (min (alength ^floats (:samples audio)) (* (inc i) 16000))))
                           (range 11))
              fin (loop [st (si m) [c & r] chunks]
                    (if c (recur (sp st c) r) (sp st (float-array 0) :final? true)))]
          (is (= jfk-text (:text fin))))))))

(deftest ^:anchors ^:gpu qwen3-asr-gpu-jfk-anchor
  ;; GPU-resident transcription (Arc/Level-Zero). Gated on weights + a ze device.
  ;; Q4K (GPU) vs Q8 (CPU) quant tiers may differ in punctuation; assert the
  ;; word content, not the exact string.
  (if-not (and (have? "Qwen3-ASR-0.6B") (.exists (java.io.File. jfk))
               (try ((requiring-resolve 'raster.gpu.ze-runtime/init!)) true
                    (catch Throwable _ false)))
    (println "SKIP qwen3-asr GPU anchor (weights or GPU not present)")
    (let [lm (requiring-resolve 'pretrained.asr.qwen3-asr/load-model)
          tr (requiring-resolve 'pretrained.asr.qwen3-asr/transcribe-gpu)
          m (lm (mdir "Qwen3-ASR-0.6B") {:gpu? true})
          words #(-> ^String % (.toLowerCase) (clojure.string/replace #"[^a-z' ]" "") (clojure.string/split #"\s+"))]
      (is (= (words jfk-text) (words (tr m jfk)))))))

(deftest ^:anchors qwen3-asr-jfk-anchor
  (if-not (and (have? "Qwen3-ASR-0.6B") (.exists (java.io.File. jfk)))
    (println "SKIP qwen3-asr anchor (weights not present)")
    (let [lm (requiring-resolve 'pretrained.asr.qwen3-asr/load-model)
          tr (requiring-resolve 'pretrained.asr.qwen3-asr/transcribe)
          m (lm (mdir "Qwen3-ASR-0.6B"))
          txt (tr m jfk {})]
      ;; gold: "And so, my fellow Americans, ask not what your country can do for
      ;; you; ask what you can do for your country." (punctuation style differs
      ;; from moonshine's — match content words)
      (is (.contains ^String txt "fellow Americans"))
      (is (.contains ^String txt "ask not what your country can do for you")))))

(deftest ^:anchors qwen3-embedding-anchor
  (if-not (have? "Qwen3-Embedding-0.6B")
    (println "SKIP qwen3-embedding anchor (weights not present)")
    (let [le (requiring-resolve 'pretrained.embed/load-embedder)
          et (requiring-resolve 'pretrained.embed/embed-texts)
          rows (requiring-resolve 'pretrained.embed/rows)
          m (le :qwen3-embedding-0.6b (mdir "Qwen3-Embedding-0.6B"))
          E (et m ["The capital of France is Paris."
                   "Paris is the capital and largest city of France."
                   "Gravitational waves are ripples in spacetime."])
          [a b c] (rows E)
          cos (fn [x y] (reduce + (map * x y)))]
      (testing "retrieval structure (validated cos 0.999 vs torch f32 gold)"
        (is (> (cos a b) 0.75) "related pair high")
        (is (< (cos a c) 0.45) "unrelated pair low")
        (is (> (- (cos a b) (cos a c)) 0.3) "clear margin")))))

(deftest ^:anchors bert-encoder-anchor
  ;; Self-contained BERT sentence-encoder: :engine :encoder -> pretrained.arch.bert
  ;; (BERT block + mean-pool + L2 over raster.dl, WordPiece; no external dependency).
  ;; CPU-only path (no device graph). Locate bge-small via the local models-dir or the
  ;; HF hub cache (~/.cache/raster/models); skip cleanly when absent.
  (let [hub-bge (str (System/getProperty "user.home")
                     "/.cache/raster/models/BAAI--bge-small-en-v1.5")
        le   (requiring-resolve 'pretrained.embed/load-embedder)
        et   (requiring-resolve 'pretrained.embed/embed-texts)
        rows (requiring-resolve 'pretrained.embed/rows)
        m (cond
            (have? "bge-small-en-v1.5")        (le :bge-small-en-v1.5 (mdir "bge-small-en-v1.5"))
            (.exists (java.io.File. hub-bge))  (le :bge-small-en-v1.5)
            :else nil)]
    (if-not m
      (println "SKIP bert-encoder anchor (bge-small weights not present)")
      (let [[a b c] (rows (et m ["The capital of France is Paris."
                                 "Paris is the capital and largest city of France."
                                 "Gravitational waves are ripples in spacetime."]))
            cos (fn [x y] (reduce + (map * x y)))]
        (testing "self-contained BERT encoder retrieval structure (bge-small, mean-pool + L2)"
          (is (= 384 (count a)) "384-d embeddings")
          (is (every? #(Float/isFinite %) (seq a)) "finite, non-degenerate")
          ;; measured: related 0.951, unrelated 0.397, margin 0.554 (thresholds have headroom)
          (is (> (cos a b) 0.85) "related pair high")
          (is (< (cos a c) 0.55) "unrelated pair low")
          (is (> (- (cos a b) (cos a c)) 0.3) "clear margin"))))))

(deftest ^:anchors ^:gpu gemma-gpu-decode-anchor
  ;; Regression guard for the resident-decode empty-graph miscompile: bind-decode!
  ;; once bound an EMPTY command graph (rms-style :fn -> nil program) -> all-zero
  ;; logits -> argmax-tie garbage tokens near vocab-max. A non-empty program +
  ;; correct greedy token would have caught it. Gated on weights + a ze device.
  (if-not (and (have? "gemma-3-270m-it") (.exists (java.io.File. (mdir "gemma-3-270m-it")))
               (gpu-up?))
    (println "SKIP gemma GPU decode anchor (weights or GPU not present)")
    (let [from (requiring-resolve 'pretrained.loader/from-pretrained)
          bind (requiring-resolve 'pretrained.decoder-gpu/bind-decode!)
          genr (requiring-resolve 'pretrained.decoder-gpu/generate-resident)
          g    (from (mdir "gemma-3-270m-it"))
          {:keys [tok encode decode]} (:tokenizer g)
          pids (vec (encode tok "The capital of France is"))
          out  (genr (bind g :maxpos 64) pids 4)]
      (is (.contains ^String (decode tok out) "Paris")
          "greedy GPU decode answers Paris (non-degenerate, non-zero logits)"))))

(deftest ^:anchors ^:gpu gemma-paged-decode-and-fork-anchor
  (if-not (and (have? "gemma-3-270m-it")
               (.exists (java.io.File. (mdir "gemma-3-270m-it")))
               (gpu-up?))
    (println "SKIP Gemma paged decode anchor (weights or GPU not present)")
    (let [from (requiring-resolve 'pretrained.loader/from-pretrained)
          bind (requiring-resolve 'pretrained.decoder-gpu/bind-decode!)
          open-paged (requiring-resolve 'pretrained.continuation.paged-decoder/open!)
          generate (requiring-resolve 'pretrained.continuation.paged-decoder/generate!)
          prime (requiring-resolve 'pretrained.continuation.paged-decoder/prime-token!)
          step (requiring-resolve 'pretrained.continuation.paged-decoder/step!)
          close-paged (requiring-resolve 'pretrained.continuation.paged-decoder/close!)
          fork (requiring-resolve 'pretrained.continuation.page-pool/fork-route!)
          route (requiring-resolve 'pretrained.continuation.page-pool/route)
          stats (requiring-resolve 'pretrained.continuation.page-pool/stats)
          buffer (requiring-resolve 'raster.gpu.core/buffer)
          close-session (requiring-resolve 'raster.gpu.core/close-session!)
          g (from (mdir "gemma-3-270m-it"))
          {:keys [tok encode decode]} (:tokenizer g)
          prompt (vec (encode tok "The capital of France is"))
          dstate (volatile! nil)
          decoder (volatile! nil)]
      (try
        (vreset! dstate (bind g :maxpos 64 :cache-mode :paged))
        (vreset! decoder (open-paged @dstate :page-size 16 :physical-pages 8))
        (let [pool (:pool @decoder)
              output (generate @decoder :base prompt 4)]
          (is (= [9079 236764 532 506] output))
          (is (.contains ^String (decode tok output) "Paris"))
          (testing "paged-only binding omits the displaced contiguous state"
            (is (nil? (buffer (:sess @dstate) :kc0)))
            (is (nil? (buffer (:sess @dstate) :vc0)))
            (is (nil? (buffer (:sess @dstate) :sc))))
          (fork pool :base :branch)
          (prime @decoder (last output))
          (let [base-next (step @decoder :base 9)]
            (prime @decoder (last output))
            (let [branch-next (step @decoder :branch 9)]
              (is (= base-next branch-next))
              (is (not= (:pages (route pool :base))
                        (:pages (route pool :branch)))
                  "a shared partial page becomes independent on append")
              (is (zero? (:active-leases (stats pool)))))))
        (finally
          (when @decoder (close-paged @decoder))
          (when @dstate (close-session (:sess @dstate))))))))

(deftest ^:anchors ^:gpu gemma-gpu-continuation-roundtrip-anchor
  (if-not (and (have? "gemma-3-270m-it") (.exists (java.io.File. (mdir "gemma-3-270m-it")))
               (gpu-up?))
    (println "SKIP gemma GPU continuation anchor (weights or GPU not present)")
    (let [from (requiring-resolve 'pretrained.loader/from-pretrained)
          bind (requiring-resolve 'pretrained.decoder-gpu/bind-decode!)
          close (requiring-resolve 'raster.gpu/close-session!)
          start (requiring-resolve 'pretrained.continuation.gpu/start-gpu)
          advance (requiring-resolve 'pretrained.continuation.gpu/advance-gpu)
          open-manager (requiring-resolve 'pretrained.continuation.manager/open-manager)
          checkpoint (requiring-resolve 'pretrained.continuation.manager/checkpoint-gpu!)
          lookup (requiring-resolve 'pretrained.continuation.manager/lookup)
          restore (requiring-resolve 'pretrained.continuation.manager/restore-gpu)
          delete-db (requiring-resolve 'datahike.api/delete-database)
          g (from (mdir "gemma-3-270m-it"))
          {:keys [tok encode]} (:tokenizer g)
          prompt (vec (encode tok "The capital of France is"))
          directory (java.nio.file.Files/createTempDirectory
                     "pretrained-gemma-kv-"
                     (make-array java.nio.file.attribute.FileAttribute 0))
          config {:store {:backend :memory :id (random-uuid)}
                  :schema-flexibility :write :keep-history? false :value-caps :default}
          cache (open-manager config directory)
          source-state (volatile! (bind g :maxpos 64))
          fresh-state (volatile! nil)]
      (try
        (let [fingerprint "gemma-3-270m-it-anchor"
              {:keys [uninterrupted-tokens first-part-tokens entry found]}
              (let [source @source-state
                    uninterrupted (advance
                                   (start source prompt {:model-fingerprint fingerprint}) 6)
                    first-part (advance
                                (start source prompt {:model-fingerprint fingerprint}) 2)
                    entry (checkpoint cache (:continuation first-part))
                    found (lookup cache fingerprint
                                  (:continuation/tokens (:continuation first-part)))]
                (close (:sess source))
                (vreset! source-state nil)
                {:uninterrupted-tokens (:tokens uninterrupted)
                 :first-part-tokens (:tokens first-part)
                 :entry entry
                 :found found})
              _ (System/gc)
              fresh (bind g :maxpos 64)
              _ (vreset! fresh-state fresh)
              restored (restore found fresh {:model-fingerprint fingerprint})
              second-part (advance restored 4)]
          (is (= (:kv/id entry) (:kv/id found))
              "Datahike resolves the exact token prefix to its mmap snapshot")
          (is (= uninterrupted-tokens
                 (into first-part-tokens (:tokens second-part)))
              "GPU -> mmap -> fresh GPU continuation is token-exact"))
        (finally
          (when-let [fresh @fresh-state]
            (close (:sess fresh)))
          (when-let [source @source-state]
            (close (:sess source)))
          (.close ^java.io.Closeable cache)
          (delete-db config)
          (with-open [paths (java.nio.file.Files/list directory)]
            (doseq [path (iterator-seq (.iterator paths))]
              (java.nio.file.Files/deleteIfExists path)))
          (java.nio.file.Files/deleteIfExists directory))))))

(deftest ^:anchors ^:gpu gpu-embedder-anchor
  ;; GPU prefill embedders (bind-embed!/embed-gpu — a DIFFERENT path than decode).
  ;; Same retrieval structure the torch-validated CPU anchor asserts.
  (let [le   (requiring-resolve 'pretrained.embed/load-embedder)
        et   (requiring-resolve 'pretrained.embed/embed-texts)
        rows (requiring-resolve 'pretrained.embed/rows)
        cos  (fn [x y] (reduce + (map * x y)))
        struct (fn [m]
                 (let [[a b c] (rows (et m ["The capital of France is Paris."
                                            "Paris is the capital and largest city of France."
                                            "Gravitational waves are ripples in spacetime."]))]
                   (is (every? #(Float/isFinite %) (seq a)) "finite, non-degenerate")
                   (is (> (cos a b) 0.75) "related high")
                   (is (< (cos a c) 0.45) "unrelated low")
                   (is (> (- (cos a b) (cos a c)) 0.3) "clear margin")))]
    (if-not (gpu-up?)
      (println "SKIP gpu-embedder anchor (GPU not present)")
      (do
        (if (have? "Qwen3-Embedding-0.6B")
          (testing "qwen3-embedding-0.6b GPU (last-token pool)"
            (struct (le :qwen3-embedding-0.6b-gpu (mdir "Qwen3-Embedding-0.6B"))))
          (println "SKIP qwen3-embedding-0.6b-gpu (weights absent)"))
        (if (have? "embeddinggemma-300m")
          (testing "embeddinggemma-300m GPU (mean pool + Dense)"
            (struct (le :embeddinggemma-300m (mdir "embeddinggemma-300m"))))
          (println "SKIP embeddinggemma-300m (weights absent)"))))))
