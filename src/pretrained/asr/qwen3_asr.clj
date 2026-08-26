(ns pretrained.asr.qwen3-asr
  "Qwen3-ASR (Qwen/Qwen3-ASR-*-hf) — 52-language ASR: AuT audio encoder + the
  standard Qwen3 decoder LM (which this stack already runs quantized).

  Pipeline: whisper-style log-mel (128 bins, slaney, torch.stft center/reflect,
  drop-last-frame, (log10+4)/4, mel time axis padded to %100) → per-1s-chunk conv
  stem (3× conv2d 3x3 s2 p1, 480ch, GELU; positions 0..12 PER CHUNK) → 18L d896
  pre-LN encoder with BLOCK-DIAGONAL attention (104-token/8s windows, dense
  bidirectional within) → projector (linear+GELU+linear → d1024) → embeddings
  spliced at <|audio_pad|> (151676) into the Qwen3 prompt → greedy decode →
  'language X<asr_text>{transcription}'.

  See .internal/qwen3_asr_port_checklist.md (agent-verified tensor names/shapes)."
  (:require [pretrained.safetensors :as st]
            [pretrained.audio :as audio]
            [pretrained.decoder :as dec]
            [pretrained.decoder-gpu :as dgpu]
            [raster.quant.pack :as qpack]
            [raster.gpu.core :as gpu]
            [raster.compiler.pipeline :as pipeline]
            [pretrained.loader :as loader]
            [pretrained.arch.qwen3 :as qwen3]
            [raster.arrays :as arr]
            [raster.dl.nn :as nn]
            [raster.dl.attention :as attn]
            [clojure.string :as str]
            [clojure.data.json :as json]))

;; ---------------------------------------------------------------------------
;; Loading: LM weights renamed to bare qwen3 descriptor names; AuT kept as-is
;; ---------------------------------------------------------------------------

(defn load-model
  "opts: :gpu? — keep the per-layer LM linears as f32 (they are otherwise
  skipped when the Q8 stream cache is hot, since CPU decode reads the streams)
  so decoder-gpu/bind-decode! can quantize them for the resident GPU graph."
  ([dir] (load-model dir {}))
  ([dir {:keys [gpu?]}]
  (let [cfg (json/read-str (slurp (str dir "/config.json")) :key-fn keyword)
        tc (:text_config cfg) ac (:audio_config cfg)
        rename #(cond
                  (str/starts-with? % "model.language_model.") (subs % 21)
                  (str/starts-with? % "model.") (subs % 6)
                  :else %)
        src (java.io.File. (str dir "/model.safetensors"))
        ;; when the Q8 stream cache is hot, skip converting the per-layer LM linears
        ;; to f32 entirely (~60% of the file) — they only feed the cached streams.
        ;; :gpu? keeps them: the GPU path quantizes from f32 at bind time.
        skip? (if (and (not gpu?)
                       (dec/stream-cache-valid? (java.io.File. (str dir "/.qstream-q8.bin")) src))
                #(re-matches #"model\.language_model\.layers\.\d+\.(self_attn\.[qkvo]_proj|mlp\.(gate|up|down)_proj)\.weight" %)
                (constantly false))
        lz (st/load-safetensors-lazy (str dir "/model.safetensors"))
        w (into {} (for [[nm _] (:tensors lz) :when (not (skip? nm))]
                     [(rename nm) (st/read-tensor lz nm)]))
        head-dim (or (:head_dim tc) (quot (:hidden_size tc) (:num_attention_heads tc)))
        lm {:config cfg :weights w :desc qwen3/descriptor :dir dir
            :n-layers (:num_hidden_layers tc) :d-model (:hidden_size tc)
            :d-ff (:intermediate_size tc) :n-q (:num_attention_heads tc)
            :n-kv (:num_key_value_heads tc) :head-dim head-dim
            :vocab (:vocab_size tc) :eps (double (:rms_norm_eps tc))
            :rope-global (double (or (get-in tc [:rope_parameters :rope_theta]) (:rope_theta tc) 1.0e6))
            :rope-local 10000.0
            :attn-scale (/ 1.0 (Math/sqrt (double head-dim)))
            :tokenizer (loader/detect-tokenizer dir)}]
    (assoc lm
           :qm (dec/quantize-stream lm :q8)
           :audio {:cfg ac
                   :d (long (:d_model ac)) :layers (long (:encoder_layers ac))
                   :heads (long (:encoder_attention_heads ac))
                   :ffn (long (:encoder_ffn_dim ac))
                   :out-dim (long (:output_dim ac))
                   :window (* 13 (quot (long (:n_window_infer ac))
                                       (* 2 (long (:n_window ac)))))}))))

(defn- t ^floats [m nm] (or (:data (get (:weights m) nm)) (throw (ex-info (str "missing " nm) {}))))

;; ---------------------------------------------------------------------------
;; Mel frontend (whisper-conventions; DFT as GEMM)
;; ---------------------------------------------------------------------------

(def ^:private N-FFT 400)
(def ^:private HOP 160)
(def ^:private N-BINS 201)
(def ^:private N-MELS 128)

(def ^:private dft-mats
  (delay (let [c (float-array (* N-BINS N-FFT)) s (float-array (* N-BINS N-FFT))
               hann (double-array N-FFT)]
           (dotimes [n N-FFT] (aset hann n (* 0.5 (- 1.0 (Math/cos (/ (* 2.0 Math/PI n) N-FFT))))))
           (dotimes [k N-BINS]
             (dotimes [n N-FFT]
               (let [a (/ (* 2.0 Math/PI k n) N-FFT)]
                 (aset c (+ (* k N-FFT) n) (float (* (aget hann n) (Math/cos a))))
                 (aset s (+ (* k N-FFT) n) (float (* (aget hann n) (- (Math/sin a))))))))
           {:c c :s s})))

(defn- hz->mel-slaney ^double [^double f]
  (if (< f 1000.0) (/ (* 3.0 f) 200.0)
      (+ 15.0 (/ (Math/log (/ f 1000.0)) (/ (Math/log 6.4) 27.0)))))

(defn- mel->hz-slaney ^double [^double m]
  (if (< m 15.0) (/ (* m 200.0) 3.0)
      (* 1000.0 (Math/exp (* (- m 15.0) (/ (Math/log 6.4) 27.0))))))

(def ^:private mel-filters
  ;; [128, 201] slaney scale + slaney norm, fmin 0 fmax 8000 @ sr 16000
  (delay (let [edges (mapv #(mel->hz-slaney (* (hz->mel-slaney 8000.0) (/ (double %) 129.0))) (range 130))
               fft-hz (mapv #(/ (* 8000.0 %) 200.0) (range N-BINS))
               w (float-array (* N-MELS N-BINS))]
           (dotimes [i N-MELS]
             (let [f0 (double (edges i)) f1 (double (edges (inc i))) f2 (double (edges (+ i 2)))
                   norm (/ 2.0 (- f2 f0))]
               (dotimes [k N-BINS]
                 (let [f (double (fft-hz k))
                       v (cond (and (>= f f0) (<= f f1)) (if (> f1 f0) (/ (- f f0) (- f1 f0)) 0.0)
                               (and (> f f1) (<= f f2)) (if (> f2 f1) (/ (- f2 f) (- f2 f1)) 0.0)
                               :else 0.0)]
                   (aset w (+ (* i N-BINS) k) (float (* norm v)))))))
           w)))

(defn log-mel
  "float PCM @16kHz → {:mel float[T,128] :T (padded to %100) :valid <real frames>}."
  [^floats pcm]
  (let [len (alength pcm)
        len (if (< len 8000) 8000 len)                    ;; min 0.5s zero-pad
        x (if (> len (alength pcm)) (java.util.Arrays/copyOf pcm (int len)) pcm)
        nf (quot len HOP)                                 ;; frames after drop-last
        pad 200
        px (fn ^double [^long i]                          ;; reflect-padded sample
             (let [j (- i pad)]
               (cond (neg? j) (aget x (- j))
                     (>= j len) (aget x (- (* 2 (dec len)) j))
                     :else (aget x j))))
        frames (float-array (* nf N-FFT))
        _ (dotimes [f nf]
            (dotimes [n N-FFT]
              (aset frames (+ (* f N-FFT) n) (float (px (+ (* f HOP) n))))))
        {:keys [c s]} @dft-mats
        zb (float-array N-BINS)
        re (nn/linear frames c zb nf N-FFT N-BINS)
        im (nn/linear frames s zb nf N-FFT N-BINS)
        pw (float-array (* nf N-BINS))
        _ (dotimes [i (* nf N-BINS)]
            (aset pw i (float (+ (* (aget ^floats re i) (aget ^floats re i))
                                 (* (aget ^floats im i) (aget ^floats im i))))))
        mel (nn/linear pw @mel-filters (float-array N-MELS) nf N-BINS N-MELS)
        ;; log10(clamp 1e-10) -> floor(max-8) -> (x+4)/4
        mx (loop [i 0 m -1.0e30]
             (if (< i (* nf N-MELS))
               (recur (inc i) (max m (Math/log10 (max 1.0e-10 (double (aget ^floats mel i)))))) m))
        floor (- mx 8.0)
        T (* 100 (quot (+ nf 99) 100))
        out (float-array (* T N-MELS))]
    (dotimes [i (* nf N-MELS)]
      (aset out i (float (/ (+ (max floor (Math/log10 (max 1.0e-10 (double (aget ^floats mel i))))) 4.0) 4.0))))
    {:mel out :T T :valid nf}))

;; ---------------------------------------------------------------------------
;; AuT conv stem: per-100-frame chunk, channels-last conv2d 3x3 s2 p1 ×3
;; ---------------------------------------------------------------------------

(defn- gelu! ^floats [^floats x]
  ;; erf-exact GELU (A&S 7.1.26) — the substrate kernel, in place.
  (nn/gelu-erf! x x (long (alength x)))
  x)

(def ^:private pe13
  ;; whisper-style sinusoidal [sin | cos], length 13, d 896
  (delay (let [d 896 half 448 pe (float-array (* 13 d))
               inc* (/ (Math/log 10000.0) (dec half))]
           (dotimes [p 13]
             (dotimes [j half]
               (let [ang (* p (Math/exp (* (- inc*) j)))]
                 (aset pe (+ (* p d) j) (float (Math/sin ang)))
                 (aset pe (+ (* p d) half j) (float (Math/cos ang))))))
           pe)))

(defn conv-stem
  "mel [T,128] (T%100=0) → tokens [ (T/100)*13, 896 ] with per-chunk PE.
  Substrate-composed: three nn/conv2d (channel-first [1,C,H,W], 3x3 stride 2
  pad 1) — torch [cout,cin,3,3] weights are used AS-IS (nn/conv2d's im2col row
  order IS the contiguous torch layout; the old channels-last path had to
  permute them at load)."
  [m ^floats mel T]
  (let [d (long (get-in m [:audio :d]))
        w1 (t m "audio_tower.conv2d1.weight")
        w2 (t m "audio_tower.conv2d2.weight")
        w3 (t m "audio_tower.conv2d3.weight")
        b1 (t m "audio_tower.conv2d1.bias") b2 (t m "audio_tower.conv2d2.bias") b3 (t m "audio_tower.conv2d3.bias")
        wout (t m "audio_tower.conv_out.weight")
        nchunks (quot (long T) 100)
        out (float-array (* nchunks 13 d))
        ^floats pe @pe13]
    (dotimes [ch nchunks]
      ;; chunk [128,100]: with cin=1, [H=128,W=100] is the same array
      ;; channel-first [1,1,128,100]
      (let [cx (float-array (* 128 100))
            _ (dotimes [f 100]
                (dotimes [b 128]
                  (aset cx (+ (* b 100) f)                 ;; [h=b, w=f]
                        (aget mel (+ (* (+ (* ch 100) f) 128) b)))))
            c1 (gelu! (nn/conv2d cx w1 b1 1 1 128 100 480 3 3 2 2 1 1))    ;; [1,480,64,50]
            c2 (gelu! (nn/conv2d c1 w2 b2 1 480 64 50 480 3 3 2 2 1 1))    ;; [1,480,32,25]
            c3 (gelu! (nn/conv2d c2 w3 b3 1 480 32 25 480 3 3 2 2 1 1))    ;; [1,480,16,13]
            ;; flatten [480,16,13] channel-first to [13, 480*16], feature = c*16 + h
            flat (float-array (* 13 7680))
            _ (dotimes [w* 13]
                (dotimes [h 16]
                  (dotimes [c 480]
                    (aset flat (+ (* w* 7680) (* c 16) h)
                          (aget ^floats c3 (+ (* c 208) (* h 13) w*))))))
            tok (nn/linear flat wout (float-array d) 13 7680 d)]
        (dotimes [i (* 13 (int d))]
          (aset out (+ (* ch 13 (int d)) i)
                (float (+ (aget ^floats tok i) (aget pe i)))))))
    {:x out :T (* nchunks 13)}))

;; ---------------------------------------------------------------------------
;; AuT encoder: 18L pre-LN (biased), MHA biased, 104-token block windows
;; ---------------------------------------------------------------------------

(defn- layer-norm-b
  "Standard biased LayerNorm over rows (raster.dl.nn/layer-norm, eps 1e-5)."
  ^floats [^floats x ^floats w ^floats b rows d]
  (nn/layer-norm x w b (long rows) (long d) 1e-5))

(defn- block-attention
  "Dense bidirectional MHA within consecutive blocks of `win` tokens — per block
  the same substrate composition the GPU path runs (scores-bidir! → softmax! →
  out!, MHA as GQA group 1). Blocks are contiguous row slices."
  ^floats [^floats q ^floats k ^floats v T heads hd win]
  (let [T (long T) heads (long heads) hd (long hd) win (long win)
        dim (* heads hd)
        scale (/ 1.0 (Math/sqrt (double hd)))
        out (float-array (* T dim))]
    (loop [b0 0]
      (when (< b0 T)
        (let [bt (min win (- T b0))
              qs (java.util.Arrays/copyOfRange q (* b0 dim) (* (+ b0 bt) dim))
              ks (java.util.Arrays/copyOfRange k (* b0 dim) (* (+ b0 bt) dim))
              vs (java.util.Arrays/copyOfRange v (* b0 dim) (* (+ b0 bt) dim))
              sc (float-array (* bt heads bt))
              ob (float-array (* bt dim))]
          (attn/attn-prefill-scores-bidir! qs ks sc bt heads 1 heads hd scale)
          (attn/attn-prefill-softmax! sc bt heads)
          (attn/attn-prefill-out! sc vs ob bt heads 1 heads hd)
          (System/arraycopy ob 0 out (* b0 dim) (* bt dim))
          (recur (+ b0 win)))))
    out))

(defn- add2! ^floats [^floats a ^floats b]
  (nn/residual-add! a b a (long (alength a))) a)

(defn aut-encode
  "Conv-stem tokens [T,896] → projected audio embeddings [T, out-dim]."
  [m ^floats x0 T]
  (let [{:keys [d layers heads ffn window out-dim]} (:audio m)
        hd (quot (long d) (long heads))
        T (long T)]
    (loop [l 0 x x0]
      (if (< l (long layers))
        (let [p (str "audio_tower.layers." l ".")
              h (layer-norm-b x (t m (str p "self_attn_layer_norm.weight")) (t m (str p "self_attn_layer_norm.bias")) T d)
              q (nn/linear h (t m (str p "self_attn.q_proj.weight")) (t m (str p "self_attn.q_proj.bias")) T d d)
              k (nn/linear h (t m (str p "self_attn.k_proj.weight")) (t m (str p "self_attn.k_proj.bias")) T d d)
              v (nn/linear h (t m (str p "self_attn.v_proj.weight")) (t m (str p "self_attn.v_proj.bias")) T d d)
              a (block-attention q k v T heads hd window)
              x1 (add2! (nn/linear a (t m (str p "self_attn.out_proj.weight")) (t m (str p "self_attn.out_proj.bias")) T d d) x)
              h2 (layer-norm-b x1 (t m (str p "final_layer_norm.weight")) (t m (str p "final_layer_norm.bias")) T d)
              f1 (gelu! (nn/linear h2 (t m (str p "fc1.weight")) (t m (str p "fc1.bias")) T d ffn))
              x2 (add2! (nn/linear f1 (t m (str p "fc2.weight")) (t m (str p "fc2.bias")) T ffn d) x1)]
          (recur (inc l) x2))
        (let [xf (layer-norm-b x (t m "audio_tower.ln_post.weight") (t m "audio_tower.ln_post.bias") T d)
              p1 (gelu! (nn/linear xf (t m "multi_modal_projector.linear_1.weight")
                                   (t m "multi_modal_projector.linear_1.bias") T d d))]
          (nn/linear p1 (t m "multi_modal_projector.linear_2.weight")
                     (t m "multi_modal_projector.linear_2.bias") T d out-dim))))))

;; ---------------------------------------------------------------------------
;; Prompt + generate (audio rows spliced into the Qwen3 decoder)
;; ---------------------------------------------------------------------------

(def ^:private AUDIO-PAD 151676)
(def ^:private IM-START 151644)
(def ^:private IM-END 151645)
(def ^:private AUDIO-START 151669)
(def ^:private AUDIO-END 151670)
(def ^:private ASR-TEXT 151704)
(def ^:private EOT 151643)

(defn- encode-text [m s]
  (let [{:keys [tok encode]} (:tokenizer m)] (vec (encode tok s))))

(declare aut-encode-gpu)

(defn- prep-prompt
  "Shared front half of transcription: mel → conv stem → AuT encode → prompt ids
  with <|audio_pad|> placeholders. Returns {:prompt ids :audio-rows floats}."
  [m wav {:keys [language context] :as opts}]
  (let [{:keys [^floats samples]} (if (string? wav) (audio/load-wav wav) wav)
        {:keys [mel T valid]} (log-mel samples)
        stem (conv-stem m mel T)
        ;; valid tokens: f(valid%100) + 13*(valid/100), f = triple (x-1)/2+1
        f1 (fn [^long x] (inc (quot (dec x) 2)))
        nvalid (+ (* 13 (quot (long valid) 100))
                  (let [r (rem (long valid) 100)] (if (pos? r) (f1 (f1 (f1 r))) 0)))
        audio-rows (if (:gpu-encoder? opts)
                     (aut-encode-gpu m (:x stem) (:T stem)
                                     (or (:device-id opts) :ze:0))
                     (aut-encode m (:x stem) (:T stem)))
        prompt (-> [IM-START]
                   (into (encode-text m (str "system\n" (or context ""))))
                   (conj IM-END) (into (encode-text m "\n")) (conj IM-START)
                   (into (encode-text m "user\n"))
                   (conj AUDIO-START)
                   (into (repeat nvalid AUDIO-PAD))
                   (conj AUDIO-END) (conj IM-END) (into (encode-text m "\n")) (conj IM-START)
                   (into (encode-text m "assistant\n"))
                   (into (if language
                           (conj (vec (encode-text m (str "language " language))) ASR-TEXT)
                           [])))]
    {:prompt prompt :audio-rows audio-rows}))

(defn- strip-asr-text [raw]
  (if-let [i (str/index-of raw "<asr_text>")]
    (subs raw (+ i 10))
    raw))

(defn transcribe
  "WAV path or {:samples} → transcription text. opts: :language (name, e.g.
  \"English\") forces the decoder prefix; :context biases via the system slot."
  ([m wav] (transcribe m wav {}))
  ([m wav {:keys [language context] :as opts}]
   (let [{:keys [prompt ^floats audio-rows]} (prep-prompt m wav opts)
         d (long (:d-model m))
         P (count prompt)
         max-new 512
         {:keys [n-layers n-kv head-dim]} m
         slot (* (+ P max-new) (long n-kv) (long head-dim))
         kc (vec (repeatedly n-layers #(float-array slot)))
         vc (vec (repeatedly n-layers #(float-array slot)))
         ;; prefill: audio positions feed the projector rows instead of embeddings
         _ (loop [p 0 ai 0]
             (when (< p P)
               (let [tok* (nth prompt p)]
                 (if (= tok* AUDIO-PAD)
                   (do (dec/decode-step m (java.util.Arrays/copyOfRange audio-rows (* ai d) (* (inc ai) d)) p kc vc)
                       (recur (inc p) (inc ai)))
                   (do (dec/decode-step m tok* p kc vc)
                       (recur (inc p) ai))))))
         ]
     ;; greedy: re-run the last prompt token (KV rewrite at dec P is idempotent)
     (let [out (loop [ids [] pos (dec P) tok* (peek prompt) n 0]
                 (let [h (dec/decode-step m tok* pos kc vc)
                       logits (dec/lm-logits m h)
                       nxt (long (arr/argmax logits))]
                   (if (or (= nxt IM-END) (= nxt EOT) (>= n max-new))
                     ids
                     (recur (conj ids nxt) (inc pos) nxt (inc n)))))
           {:keys [tok decode]} (:tokenizer m)
           raw (decode tok out)]
       (strip-asr-text raw)))))

;; ---------------------------------------------------------------------------
;; AuT encoder on GPU (phase 2b): each 104-token window attends only within
;; itself (block-diagonal) → the encoder runs as INDEPENDENT bidirectional
;; blocks of window size. One generated layer program per distinct block size
;; (full windows T=104 + one remainder size), 18 layers each, Q8 weights.
;; ---------------------------------------------------------------------------

(def ^:private aut-gen-cache (atom {}))

(defn- aut-layer-form
  "Generated AuT encoder layer deftm at block size T: biased LayerNorm →
  biased MHA (bidirectional, no rope) → residual → biased LayerNorm →
  fc1+GELU+fc2 (biased) → residual. Q8 weights (w{x}p/w{x}s) + f32 biases."
  [nm T d ffn heads]
  (let [af dgpu/af
        T (long T) d (long d) ffn (long ffn) heads (long heads)
        hd (quot d heads)
        wp2 (fn [pl] [(symbol (str "w" pl "p")) :- (af :i)
                      (symbol (str "w" pl "s")) :- (af :f)
                      (symbol (str "b" pl)) :- (af :f)])
        params (vec (concat
                     ['r-in :- (af :f)
                      'ln1g :- (af :f) 'ln1b :- (af :f)
                      'ln2g :- (af :f) 'ln2b :- (af :f)]
                     (mapcat wp2 ["q" "k" "v" "o" "f1" "f2"])
                     ['h :- (af :f) 'hp :- (af :i) 'hs :- (af :f)
                      'q :- (af :f) 'k :- (af :f) 'v :- (af :f)
                      'qb :- (af :f) 'kb :- (af :f) 'vb :- (af :f)
                      'sc :- (af :f) 'at :- (af :f)
                      'ap :- (af :i) 'as :- (af :f)
                      'o :- (af :f) 'ob :- (af :f) 'x1 :- (af :f)
                      'h2 :- (af :f) 'fp :- (af :i) 'fs :- (af :f)
                      'g1 :- (af :f) 'g1b :- (af :f) 'gel :- (af :f)
                      'gp :- (af :i) 'gs :- (af :f)
                      'dn :- (af :f) 'dnb :- (af :f)
                      'r-out :- (af :f)
                      'eps :- 'Double 'scale :- 'Double]))
        body
        [(list 'nn/layer-norm! 'r-in 'ln1g 'ln1b 'h T d 'eps)
         (list 'qk/quant-act-i8-rows-gpu! 'h 'hp 'hs d T)
         (list 'qk/qmatmul-i8-gemm! 'hp 'hs 'wqp 'wqs 'q d d T)
         (list 'qk/qmatmul-i8-gemm! 'hp 'hs 'wkp 'wks 'k d d T)
         (list 'qk/qmatmul-i8-gemm! 'hp 'hs 'wvp 'wvs 'v d d T)
         (list 'nn/add-bias-rows! 'q 'bq 'qb T d)
         (list 'nn/add-bias-rows! 'k 'bk 'kb T d)
         (list 'nn/add-bias-rows! 'v 'bv 'vb T d)
         ;; MHA: nq = nkv = heads, group 1, no rope
         (list 'attn/attn-prefill-scores-bidir! 'qb 'kb 'sc T heads 1 heads hd 'scale)
         (list 'attn/attn-prefill-softmax! 'sc T heads)
         (list 'attn/attn-prefill-out! 'sc 'vb 'at T heads 1 heads hd)
         (list 'qk/quant-act-i8-rows-gpu! 'at 'ap 'as d T)
         (list 'qk/qmatmul-i8-gemm! 'ap 'as 'wop 'wos 'o d d T)
         (list 'nn/add-bias-rows! 'o 'bo 'ob T d)
         (list 'nn/residual-add! 'r-in 'ob 'x1 (* T d))
         (list 'nn/layer-norm! 'x1 'ln2g 'ln2b 'h2 T d 'eps)
         (list 'qk/quant-act-i8-rows-gpu! 'h2 'fp 'fs d T)
         (list 'qk/qmatmul-i8-gemm! 'fp 'fs 'wf1p 'wf1s 'g1 d ffn T)
         (list 'nn/add-bias-rows! 'g1 'bf1 'g1b T ffn)
         (list 'nn/gelu-erf! 'g1b 'gel (* T ffn))   ;; erf-exact (matches CPU/torch)
         (list 'qk/quant-act-i8-rows-gpu! 'gel 'gp 'gs ffn T)
         (list 'qk/qmatmul-i8-gemm! 'gp 'gs 'wf2p 'wf2s 'dn ffn d T)
         (list 'nn/add-bias-rows! 'dn 'bf2 'dnb T d)
         (list 'nn/residual-add! 'x1 'dnb 'r-out (* T d))]]
    (list 'raster.core/deftm nm params :- 'Void (cons 'do body))))

(defn- gen-aut-layer! [m T]
  (let [{:keys [d ffn heads]} (:audio m)
        nm (symbol (str "gaut-" d "x" ffn "-t" T "!"))]
    (or (get @aut-gen-cache nm)
        (let [v (dgpu/eval-gen! nm (aut-layer-form nm T d ffn heads))]
          (swap! aut-gen-cache assoc nm v)
          v))))

(defn- aut-quantize
  "Q8-pack the 18 encoder layers' six linears. {[l pl] {:q {:wp :ws} :b bias}}.
  EAGER and memory-bounded: a plain reduce packs strictly one tensor at a time.
  (The previous chunked-lazy `for` realized a whole chunk of quantize jobs —
  every layer's packed arrays at once — before `into` consumed any, which blew
  the heap at 5-8g during the GPU anchor.)"
  [m]
  (let [{:keys [d ffn layers]} (:audio m)
        spec [["q"  "self_attn.q_proj"   d d]
              ["k"  "self_attn.k_proj"   d d]
              ["v"  "self_attn.v_proj"   d d]
              ["o"  "self_attn.out_proj" d d]
              ["f1" "fc1"                d ffn]
              ["f2" "fc2"                ffn d]]]
    (persistent!
     (reduce
      (fn [acc [l [pl suffix in out]]]
        (let [base (str "audio_tower.layers." l "." suffix)]
          (assoc! acc [l pl]
                  {:q (qpack/quantize-one-q8 (t m (str base ".weight")) in out base)
                   :b (t m (str base ".bias"))})))
      (transient {})
      (vec (for [l (range layers) s spec] [l s]))))))

(defn- bind-aut!
  "Bind the 18-layer AuT block program at block size T into a fresh session.
  Returns {:sess :T}. Cached per T on the model's ::aut-sessions atom is the
  caller's business — binding is ~seconds (program compile is cached by name)."
  [m T device-id]
  (let [{:keys [d ffn layers heads]} (:audio m)
        T (long T) d (long d) ffn (long ffn)
        hd (quot d heads)
        layer-var (gen-aut-layer! m T)
        prog (pipeline/compile-gpu-program layer-var device-id :dtype :float)
        qz (aut-quantize m)
        scratch {:h [:float (* T d) nil] :hp [:int (* T (quot d 4)) nil]
                 :hs [:float (* T (quot d 32)) nil]
                 :q [:float (* T d) nil] :k [:float (* T d) nil] :v [:float (* T d) nil]
                 :qb [:float (* T d) nil] :kb [:float (* T d) nil] :vb [:float (* T d) nil]
                 :sc [:float (* T heads T) nil] :at [:float (* T d) nil]
                 :ap [:int (* T (quot d 4)) nil] :as [:float (* T (quot d 32)) nil]
                 :o [:float (* T d) nil] :ob [:float (* T d) nil] :x1 [:float (* T d) nil]
                 :h2 [:float (* T d) nil] :fp [:int (* T (quot d 4)) nil]
                 :fs [:float (* T (quot d 32)) nil]
                 :g1 [:float (* T ffn) nil] :g1b [:float (* T ffn) nil]
                 :gel [:float (* T ffn) nil]
                 :gp [:int (* T (quot ffn 4)) nil] :gs [:float (* T (quot ffn 32)) nil]
                 :dn [:float (* T d) nil] :dnb [:float (* T d) nil]}
        residuals (into {} (for [l (range (inc layers))]
                             [(keyword (str "ar" l)) [:float (* T d) nil]]))
        weights (into {}
                  (mapcat (fn [[[l pl] {:keys [q b]}]]
                            [[(keyword (str "AL" l "w" pl "p")) [:int (alength ^ints (:wp q)) (:wp q)]]
                             [(keyword (str "AL" l "w" pl "s")) [:float (alength ^floats (:ws q)) (:ws q)]]
                             [(keyword (str "AL" l "b" pl)) [:float (alength ^floats b) b]]])
                          qz))
        norms (into {}
                (mapcat (fn [l]
                          (let [p (str "audio_tower.layers." l ".")]
                            [[(keyword (str "AL" l "ln1g")) [:float d (t m (str p "self_attn_layer_norm.weight"))]]
                             [(keyword (str "AL" l "ln1b")) [:float d (t m (str p "self_attn_layer_norm.bias"))]]
                             [(keyword (str "AL" l "ln2g")) [:float d (t m (str p "final_layer_norm.weight"))]]
                             [(keyword (str "AL" l "ln2b")) [:float d (t m (str p "final_layer_norm.bias"))]]]))
                        (range layers)))
        args ((deref #'dgpu/scalar-args) prog {"eps" 1e-5 "scale" (/ 1.0 (Math/sqrt (double hd)))})
        prog-allocs (into {}
                      (for [l (range layers)
                            {:keys [sym size-fn]} (:allocs prog)]
                        [(keyword (str "AL" l "_" (name sym))) [:float (long (size-fn args)) nil]]))
        sess (gpu/make-session device-id)
        pbk (fn [l] (fn [a]
                      (let [pn (name a)]
                        (cond (= pn "r-in")  (keyword (str "ar" l))
                              (= pn "r-out") (keyword (str "ar" (inc l)))
                              (or (= \w (first pn)) (= \b (first pn))
                                  (str/starts-with? pn "ln"))
                              (keyword (str "AL" l pn))
                              (re-find #"__rbuf" pn) (keyword (str "AL" l "_" pn))
                              :else (keyword pn)))))
        phases (atom [])]
    (gpu/alloc! sess (merge scratch residuals weights norms prog-allocs))
    (doseq [l (range layers)]
      (doseq [step (:steps prog)]
        (let [ph (keyword (str "AL" l "_" (name (:phase step))))]
          (gpu/bind-step! sess (assoc step :phase ph) args (pbk l))
          (swap! phases conj ph))))
    (gpu/record-graph! sess @phases :aut)
    {:sess sess :T T :device-id device-id}))

(def ^:private aut-sessions (atom {}))

(defn aut-encode-gpu
  "GPU AuT encode: run each `win`-sized block through the resident 18-layer
  program (one session per distinct block size, cached), then ln_post +
  projector on the CPU (tiny). Drop-in for aut-encode."
  (^floats [m ^floats x0 T] (aut-encode-gpu m x0 T :ze:0))
  (^floats [m ^floats x0 T device-id]
   (let [{:keys [d layers window out-dim]} (:audio m)
         T (long T) d (long d) win (long window)
         nblk (quot (+ T (dec win)) win)
         out (float-array (* T d))]
     (dotimes [bi nblk]
       (let [b0 (* bi win)
             bt (min win (- T b0))
             {:keys [sess]} (or (get @aut-sessions [(:dir m) bt device-id])
                                (let [s (bind-aut! m bt device-id)]
                                  (swap! aut-sessions assoc [(:dir m) bt device-id] s) s))]
         (gpu/upload! sess :ar0 (java.util.Arrays/copyOfRange x0 (* b0 d) (* (+ b0 bt) d)))
         (gpu/replay! sess :aut)
         (System/arraycopy (gpu/download sess (keyword (str "ar" layers))) 0
                           out (* b0 d) (* bt d))))
     ;; final norm + projector (CPU: [T,d] x 2 small GEMMs)
     (let [xf (layer-norm-b out (t m "audio_tower.ln_post.weight")
                            (t m "audio_tower.ln_post.bias") T d)
           p1 (gelu! (nn/linear xf (t m "multi_modal_projector.linear_1.weight")
                                (t m "multi_modal_projector.linear_1.bias") T d d))]
       (nn/linear p1 (t m "multi_modal_projector.linear_2.weight")
                  (t m "multi_modal_projector.linear_2.bias") T d out-dim)))))

;; ---------------------------------------------------------------------------
;; GPU-resident transcription (Level Zero/OpenCL via pretrained.decoder-gpu)
;; ---------------------------------------------------------------------------

(defn bind-gpu
  "Compile + bind the resident GPU decode graph for this ASR model's Qwen3 LM.
  Expensive (compile + Q-quantize + upload); do it once and reuse the returned
  dstate across transcriptions. maxpos bounds prompt+generation length."
  [m & {:keys [maxpos prefill-T device-id]
        :or {maxpos 1024 device-id :ze:0}}]
  ;; prefill-T (opt-in): binds the batched-prefill graph. MEASURED on Arc/ze:
  ;; one prefill replay (28 layers x ~15 steps = 420 kernels) is LATENCY-bound
  ;; at these dims — 4.9s @T=256 vs ~1.2s for 158 sequential decode replays —
  ;; so per-token priming stays the default until the graph runs at higher
  ;; occupancy (fused steps or bigger batch). Correctness is validated.
  (dgpu/bind-decode! m :maxpos maxpos :prefill-T prefill-T
                       :device-id device-id))

(defn transcribe-gpu
  "GPU-resident transcription: the prompt primes the KV cache one row per graph
  replay — token positions feed embedding rows, <|audio_pad|> positions feed the
  AuT projector rows (the multimodal seam is just decode-row!) — then the
  autoregressive rollout runs fully on-device (argmax + next-embed in the graph
  tail; per-token host traffic is position metadata up and a 4-byte token id down).
  opts: :language, :context (as transcribe), :dstate (reuse a bind-gpu result),
  :max-new (default 512)."
  ([m wav] (transcribe-gpu m wav {}))
  ([m wav {:keys [dstate max-new] :or {max-new 512} :as opts}]
   (let [opts (merge {:gpu-encoder? true} opts)   ; GPU AuT unless disabled
         {:keys [prompt ^floats audio-rows]} (prep-prompt m wav opts)
         P (count prompt)
         d (long (:d-model m))
         dstate (or dstate (bind-gpu m :maxpos (+ P (long max-new))
                                      :device-id (or (:device-id opts) :ze:0)))
         maxpos (long (:maxpos dstate))]
     (when (> (+ P (long max-new)) maxpos)
       (throw (ex-info (str "prompt (" P ") + max-new (" max-new
                            ") exceeds the bound dstate's maxpos " maxpos
                            " — bind-gpu with a larger :maxpos")
                       {:P P :max-new max-new :maxpos maxpos})))
     ;; fill the KV cache for the prompt: BATCHED when the dstate carries a
     ;; prefill graph (one replay over all positions, phase 2a), else one
     ;; graph replay per position (phase 1). In the batched path the last
     ;; position is re-primed through the decode graph so the tail leaves
     ;; the first generated token + its embedding on-device, as before.
     (if-let [Tp (:prefill-T dstate)]
       (do (when (> P (long Tp))
             (throw (ex-info (str "prompt " P " exceeds prefill-T " Tp) {})))
           (let [rows (float-array (* (long Tp) d))]
             (loop [p 0 ai 0]
               (when (< p P)
                 (let [tok* (nth prompt p)]
                   (if (= tok* AUDIO-PAD)
                     (do (System/arraycopy audio-rows (* ai d) rows (* p d) d)
                         (recur (inc p) (inc ai)))
                     (do (System/arraycopy (dgpu/embed-row m tok*) 0 rows (* p d) d)
                         (recur (inc p) ai))))))
             (dgpu/prefill-rows! dstate rows)
             ;; re-prime the last position through the decode graph (idempotent
             ;; KV rewrite) so the tail seeds tokbuf/r0 for the rollout
             (dgpu/decode-token! dstate (peek prompt) (dec P))))
       (loop [p 0 ai 0]
         (when (< p P)
           (let [tok* (nth prompt p)]
             (if (= tok* AUDIO-PAD)
               (do (dgpu/decode-row! dstate (java.util.Arrays/copyOfRange
                                             audio-rows (* ai d) (* (inc ai) d)) p)
                   (recur (inc p) (inc ai)))
               (do (dgpu/decode-token! dstate tok* p)
                   (recur (inc p) ai)))))))
     (let [{:keys [sess]} dstate
           t0 (aget ^ints (gpu/download sess :tokbuf) 0)
           out (if (or (= t0 IM-END) (= t0 EOT))
                 []
                 (loop [p P out [t0]]
                   (if (and (< (count out) (long max-new)) (< p maxpos))
                     (let [t (dgpu/resident-step! dstate p)]
                       (if (or (= t IM-END) (= t EOT))
                         out
                         (recur (inc p) (conj out t))))
                     out)))
           {:keys [tok decode]} (:tokenizer m)]
       (strip-asr-text (decode tok out))))))
