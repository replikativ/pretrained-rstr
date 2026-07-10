(ns pretrained.asr.moonshine
  "Moonshine v2 streaming ASR (UsefulSensors/moonshine-streaming-*) — CPU f32 port.

  Encoder-decoder over raw 16kHz PCM (no mel): per-frame CMVN + asinh compression
  + linear + two causal stride-2 convs → 50Hz×768 features; 14-layer bidirectional
  encoder with per-layer sliding windows and NO positional embeddings ('ergodic' —
  what makes true streaming possible); adapter adds learned absolute positions and
  projects 768→640; 14-layer causal decoder with partial-interleaved RoPE self-attn,
  cross-attention to the adapter memory, and chunked SwiGLU. Greedy decode.

  Port traps (validated against HF activations): encoder LayerNorm scale = γ+1.0;
  SwiGLU gate = SECOND half of fc1; GELU is erf-exact; RoPE rotates only 32/64 dims
  in GPT-J interleaved pairing; sliding window right=4 admits only 3 future frames.
  See .internal/moonshine_port_checklist.md."
  (:require [pretrained.safetensors :as st]
            [pretrained.audio :as audio]
            [raster.core :refer [deftm]]
            [raster.arrays :as arr]
            [raster.numeric :as num]
            [raster.math :as rmath]
            [raster.dl.nn :as nn]
            [raster.dl.attention :as attn]
            [pretrained.decoder :as dec]
            [raster.linalg.blas :as blas]
            [raster.quant.op :as ql]
            [raster.compiler.backend.cpu.quant :as cq]
            [clojure.string :as str]
            [clojure.data.json :as json]))

;; ---------------------------------------------------------------------------
;; Loading
;; ---------------------------------------------------------------------------

(defn load-model [dir]
  (let [cfg (json/read-str (slurp (str dir "/config.json")) :key-fn keyword)
        w (st/load-safetensors (str dir "/model.safetensors"))
        vocab (get (json/read-str (slurp (str dir "/tokenizer.json"))) "model")
        id->tok (reduce-kv (fn [m t i] (assoc m (long i) t)) {} (get vocab "vocab"))
        ;; added tokens (specials + ST_*) override
        id->tok (reduce (fn [m at] (assoc m (long (get at "id")) (get at "content")))
                        id->tok
                        (get (json/read-str (slurp (str dir "/tokenizer.json"))) "added_tokens"))]
    ;; decoder matvecs run int8 (Q8_0, transcription-lossless) on the spin-pool
    ;; int8-MAC kernel — the same fast path as the decoder-LM engine. Encoder stays
    ;; f32 GEMM (batched, compute-dense).
    {:cfg cfg :w w :id->tok id->tok
     :qw (let [src (java.io.File. (str dir "/model.safetensors"))
               cf (java.io.File. (str dir "/.qstream-q8.bin"))]
           (if (dec/stream-cache-valid? cf src)
             (@#'dec/read-stream-cache cf)
             (let [qw (into {} (pmap (fn [nm]
                                       (let [{:keys [data shape]} (get w nm)
                                             [out in] shape
                                             {:keys [wq ws]} (cq/quantize-weight data cq/q8-0)]
                                         [nm (assoc (cq/repack-stream wq ws (long out) (long in) cq/q8-0)
                                                    :in (long in) :out (long out))]))
                                     (concat ["proj_out.weight"]
                                             (for [l (range 14)
                                                   nm ["self_attn.q_proj.weight" "self_attn.k_proj.weight"
                                                       "self_attn.v_proj.weight" "self_attn.o_proj.weight"
                                                       "encoder_attn.q_proj.weight" "encoder_attn.o_proj.weight"
                                                       "mlp.fc1.weight" "mlp.fc2.weight"]]
                                               (str "model.decoder.layers." l "." nm)))))]
               (@#'dec/write-stream-cache! cf src qw)
               qw)))
     :enc-layers (long (get-in cfg [:encoder_config :num_hidden_layers]))
     :dec-layers (long (:num_hidden_layers cfg))
     :d-enc (long (get-in cfg [:encoder_config :hidden_size]))
     :d-dec (long (:hidden_size cfg))
     :heads (long (:num_attention_heads cfg))
     :hd (long (:head_dim cfg))
     :windows (mapv (fn [[l r]] [(long l) (long r)])
                    (get-in cfg [:encoder_config :sliding_windows]))}))

(defn- t ^floats [m nm] (or (:data (get (:w m) nm)) (throw (ex-info (str "missing tensor " nm) {}))))

;; ---------------------------------------------------------------------------
;; Primitives (CPU f32 reference)
;; ---------------------------------------------------------------------------

(def ^:private ^floats zero640 (float-array 640))
(def ^:private ^floats zero768 (float-array 768))
(def ^:private ^floats zero1536 (float-array 1536))

(defn- zeros ^floats [n] (case (int n) 640 zero640 768 zero768 1536 zero1536 (float-array n)))

(defn- layer-norm!
  "In-place-ish LayerNorm over rows: mean-centered, eps 1e-5, scale (+gain-offset w),
  no bias. Returns a NEW array."
  ^floats [^floats x ^floats w gain-offset rows d]
  (let [gain-offset (double gain-offset) rows (long rows) d (long d)
        out (float-array (* rows d))]
    (dotimes [r rows]
      (let [base (* r d)
            mean (/ (loop [i 0 s 0.0] (if (< i d) (recur (inc i) (+ s (aget x (+ base i)))) s)) d)
            var (/ (loop [i 0 s 0.0]
                     (if (< i d)
                       (recur (inc i) (let [v (- (aget x (+ base i)) mean)] (+ s (* v v))))
                       s)) d)
            inv (/ 1.0 (Math/sqrt (+ var 1e-5)))]
        (dotimes [i d]
          (aset out (+ base i)
                (float (* (- (aget x (+ base i)) mean) inv (+ gain-offset (aget w i))))))))
    out))

(defn- gelu-erf! ^floats [^floats x]
  ;; erf-exact GELU (A&S 7.1.26) — the substrate kernel, in place.
  (nn/gelu-erf! x x (long (alength x)))
  x)

(defn- linear1
  "Batch-1 linear y = W@x + b via sgemv (no GEMM thread-spawn overhead)."
  ^floats [^floats x ^floats w ^floats b out in]
  (let [y (java.util.Arrays/copyOf b (int out))]
    (blas/dgemv! w x y (long out) (long in) (float 1.0) (float 1.0))
    y))

(defn- qlin
  "Quantized decode matvec: y = W@x (+ b) via the int8-MAC spin-pool kernel."
  ^floats [m nm ^floats x ^floats b]
  (let [{:keys [wqi wsi in out]} (get (:qw m) nm)
        ^floats y (ql/qlinear-i8-q8 x wqi wsi in out)]
    (when b (nn/add-bias-rows! y b y 1 (long (alength y))))
    y))

(defn- add! ^floats [^floats a ^floats b]
  (nn/residual-add! a b a (long (alength a)))
  a)

;; ---------------------------------------------------------------------------
;; Frontend: raw PCM → 50Hz × 768
;; ---------------------------------------------------------------------------

(deftm cmvn-asinh!
  "Per-80-sample-frame CMVN (mean, biased variance, eps 1e-6) + asinh(kk·x)
  compression, FUSED: the normalized value stays a double all the way into
  raster.math/asinh, with a single f32 rounding at the store. The streaming
  invariant (finalized encoder frames bit-identical between streaming and
  batch, validated vs HF torch) forbids splitting this into a layer-norm pass
  + an asinh pass — that would round the intermediate to f32 and shift ULPs.
  rmath/asinh's double path is the identical log(y + sqrt(y²+1)) formula.
  The explicit (double ...) casts are load-bearing: in a float-array kernel
  the walker otherwise monomorphizes the scalar accumulation to f32 (literal
  promotion), which loses the f64 accumulation the torch gold requires —
  verified 1-ULP drift without them, byte-identical with them."
  [pcm :- (Array float) out :- (Array float) n80 :- Long kk :- Double] :- Void
  (dotimes [f n80]
    (let [base (* f 80)
          mean (num//
                (loop [i 0 s 0.0]
                  (if (< i 80)
                    (recur (inc i) (num/+ (double s) (double (arr/aget pcm (+ base i)))))
                    s))
                80.0)
          var (num//
               (loop [i 0 s 0.0]
                 (if (< i 80)
                   (recur (inc i)
                          (let [v (num/- (double (arr/aget pcm (+ base i))) (double mean))]
                            (num/+ (double s) (num/* v v))))
                   s))
               80.0)
          inv (num// (double 1.0) (double (num/sqrt (num/+ (double var) 1.0e-6))))]
      (dotimes [i 80]
        (let [v (num/* (num/- (double (arr/aget pcm (+ base i))) (double mean)) (double inv))]
          (arr/aset out (+ base i) (float (rmath/asinh (num/* (double kk) v)))))))))

(defn frontend
  "Raw float PCM (multiple of 80 samples) → features [T,768] row-major + T.
  Substrate-composed: cmvn-asinh! (above) → linear+silu → two causal
  stride-2 channels-last convs (nn/conv1d-cl, pad-left k-1)."
  [m ^floats pcm]
  (let [n80 (quot (alength pcm) 80)
        framed (float-array (* n80 80))
        log-k (aget ^floats (t m "model.encoder.embedder.comp.log_k") 0)
        kk (Math/exp (double log-k))]
    (cmvn-asinh! pcm framed n80 kk)
    (let [h (nn/silu (nn/linear framed (t m "model.encoder.embedder.linear.weight")
                                (zeros 768) n80 80 768)
                     (* n80 768))
          T1 (inc (quot (dec n80) 2))
          c1 (nn/silu (nn/conv1d-cl h (t m "model.encoder.embedder.conv1.weight")
                                    (t m "model.encoder.embedder.conv1.bias")
                                    n80 768 1536 5 2 4 0)
                      (* T1 1536))
          c2 (nn/conv1d-cl c1 (t m "model.encoder.embedder.conv2.weight")
                           (t m "model.encoder.embedder.conv2.bias")
                           T1 1536 768 5 2 4 0)]
      {:x c2 :T (inc (quot (dec T1) 2))})))

;; ---------------------------------------------------------------------------
;; Encoder: 14 layers, sliding-window bidirectional MHA, no positions
;; ---------------------------------------------------------------------------

(defn- windowed-attention
  "MHA with sliding window [left right]: query i attends j where
  (0 <= i-j < left) or (0 < j-i < right). q,k,v [T, heads*hd]; returns [T, heads*hd].
  Substrate composition (MHA as GQA group 1): windowed scores (out-of-window =
  -1e30, exp underflows to exact 0 under the shared softmax) → softmax → out."
  ^floats [^floats q ^floats k ^floats v T heads hd left right]
  (let [T (long T) heads (long heads) hd (long hd)
        dim (* heads hd)
        sc (float-array (* T heads T))
        out (float-array (* T dim))]
    (attn/attn-prefill-scores-windowed! q k sc T heads 1 heads hd
                                        (/ 1.0 (Math/sqrt (double hd)))
                                        (long left) (long right))
    (attn/attn-prefill-softmax! sc T heads)
    (attn/attn-prefill-out! sc v out T heads 1 heads hd)
    out))

(defn encode
  "Frontend features [T,768] → encoder output [T,768] (final-normed)."
  [m ^floats feats T]
  (let [T (long T) d (long (:d-enc m)) heads (long (:heads m)) hd (long (:hd m))
        adim (* heads hd)]
    (loop [l 0 x feats]
      (if (< l (:enc-layers m))
        (let [p (str "model.encoder.layers." l ".")
              [left right] (nth (:windows m) l)
              h (layer-norm! x (t m (str p "input_layernorm.gamma")) 1.0 T d)
              q (nn/linear h (t m (str p "self_attn.q_proj.weight")) (zeros adim) T d adim)
              k (nn/linear h (t m (str p "self_attn.k_proj.weight")) (zeros adim) T d adim)
              v (nn/linear h (t m (str p "self_attn.v_proj.weight")) (zeros adim) T d adim)
              a (windowed-attention q k v T heads hd left right)
              x (add! (nn/linear a (t m (str p "self_attn.o_proj.weight")) (zeros d) T adim d) x)
              h2 (layer-norm! x (t m (str p "post_attention_layernorm.gamma")) 1.0 T d)
              f1 (gelu-erf! (nn/linear h2 (t m (str p "mlp.fc1.weight"))
                                       (t m (str p "mlp.fc1.bias")) T d 3072))
              x (add! (nn/linear f1 (t m (str p "mlp.fc2.weight"))
                                 (t m (str p "mlp.fc2.bias")) T 3072 d) x)]
          (recur (inc l) x))
        (layer-norm! x (t m "model.encoder.final_norm.gamma") 1.0 T d)))))

;; ---------------------------------------------------------------------------
;; Adapter: + learned abs positions, project 768→640
;; ---------------------------------------------------------------------------

(defn adapter
  "Encoder output [T,768] (+pos offset) → memory [T,640]."
  [m ^floats enc T pos-offset]
  (let [T (long T) d (long (:d-enc m)) dd (long (:d-dec m))
        pe (t m "model.decoder.pos_emb.weight")
        xp (float-array (* T d))]
    (dotimes [i (* T d)] (aset xp i (aget enc i)))
    (dotimes [ti T]
      (let [pb (* (+ (long pos-offset) ti) d)]
        (dotimes [j d]
          (aset xp (+ (* ti d) j) (float (+ (aget xp (+ (* ti d) j)) (aget pe (+ pb j))))))))
    (nn/linear xp (t m "model.decoder.proj.weight") (zeros dd) T d dd)))

;; ---------------------------------------------------------------------------
;; Decoder: causal self-attn (partial interleaved RoPE) + cross-attn + SwiGLU
;; ---------------------------------------------------------------------------

(def ^:private max-positions
  "Preallocated KV-cache rows — the decode loops cap tokens at (min 256 ...)."
  256)

(defn- attend-cache
  "Single-query MHA over cached K/V [n, heads*hd] — MHA as GQA with n-kv = n-q
  (group 1) on the raster substrate decode-attention kernel."
  ^floats [^floats q ^floats kc ^floats vc n heads hd]
  (attn/gqa-decode-attention q kc vc (long n) (long heads) (long heads) (long hd)
                             (/ 1.0 (Math/sqrt (double hd)))))

(defn- attend-cache+w
  "attend-cache that also ACCUMULATES head-averaged attention weights into wsink
  (float[n]) — cross-attention alignment for timestamps (substrate kernel)."
  ^floats [^floats q ^floats kc ^floats vc n heads hd ^floats wsink]
  (attn/gqa-decode-attention-weights! q kc vc (long n) (long heads) (long heads) (long hd)
                                      (/ 1.0 (Math/sqrt (double hd))) wsink))

(defn- decoder-state [m ^floats memory T]
  (let [dd (long (:d-dec m))]
    {:memory memory :T (long T)
     ;; cross K/V per layer, computed once per memory version
     :cross (mapv (fn [l]
                    (let [p (str "model.decoder.layers." l ".encoder_attn.")]
                      {:k (nn/linear memory (t m (str p "k_proj.weight")) (zeros dd) T dd dd)
                       :v (nn/linear memory (t m (str p "v_proj.weight")) (zeros dd) T dd dd)}))
                  (range (:dec-layers m)))
     ;; self K/V: preallocated [max-positions, dd] row-major caches, appended
     ;; in place at each decode position (the decoder.clj decode-step pattern)
     :self-k (mapv (fn [_] (float-array (* max-positions dd))) (range (:dec-layers m)))
     :self-v (mapv (fn [_] (float-array (* max-positions dd))) (range (:dec-layers m)))}))

(defn- decode-step
  "One greedy decoder step: token id + position → next logits argmax. The 5-arg
  arity accumulates layer+head-averaged CROSS-attention weights into wsink
  (float[T]) for timestamp alignment."
  ([m state id pos] (decode-step m state id pos nil))
  ([m state id pos ^floats wsink]
  (let [dd (long (:d-dec m)) heads (long (:heads m)) hd (long (:hd m))
        emb (t m "model.decoder.embed_tokens.weight")
        x (java.util.Arrays/copyOfRange emb (* (long id) dd) (* (inc (long id)) dd))]
    (loop [l 0 ^floats x x]
      (if (< l (:dec-layers m))
        (let [p (str "model.decoder.layers." l ".")
              ;; self-attention
              h (layer-norm! x (t m (str p "input_layernorm.weight")) 0.0 1 dd)
              ;; partial interleaved RoPE: rotate first 32 of each 64-dim head (GPT-J
              ;; pairing, NOT the NeoX half-split — the moonshine port trap)
              q (attn/rope-pos-partial! (qlin m (str p "self_attn.q_proj.weight") h nil)
                                        heads hd 32 10000.0 pos)
              k (attn/rope-pos-partial! (qlin m (str p "self_attn.k_proj.weight") h nil)
                                        heads hd 32 10000.0 pos)
              v (qlin m (str p "self_attn.v_proj.weight") h nil)
              ^floats kcl (nth (:self-k state) l) ^floats vcl (nth (:self-v state) l)
              _ (attn/kv-append! k kcl dd pos)
              _ (attn/kv-append! v vcl dd pos)
              a (attend-cache q kcl vcl (inc pos) heads hd)
              x (add! (qlin m (str p "self_attn.o_proj.weight") a nil) x)
              ;; cross-attention
              h2 (layer-norm! x (t m (str p "post_attention_layernorm.weight")) 0.0 1 dd)
              q2 (qlin m (str p "encoder_attn.q_proj.weight") h2 nil)
              {ck :k cv :v} (nth (:cross state) l)
              ;; alignment signal lives in the LATER layers (early ones attend
              ;; diffusely — the whisper alignment-heads lesson)
              a2 (if (and wsink (>= l 7))
                   (attend-cache+w q2 ck cv (:T state) heads hd wsink)
                   (attend-cache q2 ck cv (:T state) heads hd))
              x (add! (qlin m (str p "encoder_attn.o_proj.weight") a2 nil) x)
              ;; SwiGLU MLP: fc1 → [value | gate], silu(gate)*value, fc2
              h3 (layer-norm! x (t m (str p "final_layernorm.weight")) 0.0 1 dd)
              f1 (qlin m (str p "mlp.fc1.weight") h3 (t m (str p "mlp.fc1.bias")))
              gv (float-array 2560)
              _ (dotimes [i 2560]
                  (let [value (aget ^floats f1 i)
                        gate (double (aget ^floats f1 (+ 2560 i)))]
                    (aset gv i (float (* value (/ gate (+ 1.0 (Math/exp (- gate)))))))))
              x (add! (qlin m (str p "mlp.fc2.weight") gv (t m (str p "mlp.fc2.bias"))) x)]
          (recur (inc l) x))
        (let [xf (layer-norm! x (t m "model.decoder.norm.weight") 0.0 1 dd)
              logits (qlin m "proj_out.weight" xf nil)]
          (long (arr/argmax logits))))))))

;; ---------------------------------------------------------------------------
;; Tokenizer decode (SP-Llama: ▁→space, byte-fallback) + top-level API
;; ---------------------------------------------------------------------------

(defn decode-tokens [m ids]
  (let [sb (java.io.ByteArrayOutputStream.)]
    (doseq [id ids]
      (let [tok (get (:id->tok m) (long id) "")]
        (if-let [[_ hex] (re-matches #"<0x([0-9A-Fa-f]{2})>" tok)]
          (.write sb (int (Long/parseLong hex 16)))
          (when-not (#{"<s>" "</s>" "<unk>"} tok)
            (.write sb (.getBytes (.replace ^String tok "▁" " ") "UTF-8"))))))
    (let [s (String. (.toByteArray sb) "UTF-8")]
      (if (.startsWith s " ") (subs s 1) s))))


;; ---------------------------------------------------------------------------
;; Streaming: ergodicity = bounded receptive fields, so recomputing a trailing
;; window reproduces finalized frames bit-identically (no conv/encoder state).
;; Frontend frame t needs 200Hz frames [4t-12, 4t] (samples 320t-960..320t+79);
;; encoder frame t needs <=16 past frames per layer -> 224 total. Finalization
;; withholds 16 frames (4 lookahead layers x window 4 = 320ms) until :final?.
;; ---------------------------------------------------------------------------

(defn- cat-blocks
  "Concatenate variable-length float[] blocks."
  ^floats [blocks]
  (let [total (reduce + 0 (map #(alength ^floats %) blocks))
        out (float-array total)]
    (loop [off 0 [b & r] blocks]
      (if b
        (do (System/arraycopy ^floats b 0 out off (alength ^floats b))
            (recur (+ off (alength ^floats b)) r))
        out))))

(defn stream-init
  "Streaming session state. Push PCM with stream-push!; finish with :final? true."
  [m]
  {:m m :samples [] :nsamples 0 :feats [] :nfeats 0 :emitted 0 :memory [] :ids [1]})

(defn- frontend-new-frames
  "Compute feature frames [nfeats, Ttot) from accumulated samples, via a trailing
  recompute window starting 3 output frames early (dropping the boundary frames)."
  [m ^floats all nfeats]
  (let [n80 (quot (alength all) 80)
        Ttot (inc (quot (dec (inc (quot (dec n80) 2))) 2))]
    (if (<= Ttot (long nfeats))
      {:new nil :Ttot Ttot}
      (let [a (long nfeats)
            f0 (max 0 (- a 3))                         ;; 3-frame guard = 12 200Hz frames
            k0 (* 4 f0)                                ;; aligned 200Hz frame start
            slice (java.util.Arrays/copyOfRange all (* 80 k0) (* n80 80))
            {:keys [^floats x T]} (frontend m slice)
            drop-n (- a f0)                            ;; boundary frames to discard
            cnt (- Ttot a)]
        {:new (java.util.Arrays/copyOfRange x (* drop-n 768)
                                            (* (+ drop-n cnt) 768))
         :Ttot Ttot}))))

(defn stream-push!
  "Feed a PCM chunk (float[] @16kHz). Returns updated state; when :decode? is set
  (or :final?) also re-decodes and attaches :text. Finalized encoder frames are
  bit-identical to batch processing (verified)."
  [state ^floats chunk & {:keys [decode? final?]}]
  (let [m (:m state)
        samples (conj (:samples state) chunk)
        nsamples (+ (long (:nsamples state)) (alength chunk))
        all (float-array nsamples)
        _ (loop [off 0 [c & r] samples]
            (when c
              (System/arraycopy ^floats c 0 all off (alength ^floats c))
              (recur (+ off (alength ^floats c)) r)))
        all (java.util.Arrays/copyOfRange all 0 (* 80 (quot nsamples 80)))
        {:keys [^floats new Ttot]} (frontend-new-frames m all (:nfeats state))
        feats (if new (conj (:feats state) new) (:feats state))
        nfeats (if new Ttot (:nfeats state))
        stable (if final? (long nfeats) (max 0 (- (long nfeats) 16)))
        emitted (long (:emitted state))
        state (assoc state :samples samples :nsamples nsamples :feats feats :nfeats nfeats)
        state (if (> stable emitted)
                (let [e0 (max 0 (- emitted 224))
                      featarr (cat-blocks feats)     ;; chunks are row blocks
                      fslice (java.util.Arrays/copyOfRange featarr (* (long e0) 768) (* (long nfeats) 768))
                      Ts (- (long nfeats) (long e0))
                      encs (encode m fslice Ts)
                      new-final (java.util.Arrays/copyOfRange encs (* (- emitted (long e0)) 768)
                                                              (* (- stable (long e0)) 768))
                      memnew (adapter m new-final (- stable emitted) emitted)]
                  (assoc state :memory (conj (:memory state) memnew) :emitted stable))
                state)
        state (if (and (or decode? final?) (pos? (long (:emitted state))))
                (let [mem (cat-blocks (:memory state))
                      T (long (:emitted state))
                      dstate (decoder-state m mem T)
                      max-tokens (min 256 (long (Math/ceil (* 6.5 (/ nsamples 16000.0)))))
                      ids (loop [ids [1] pos 0]
                            (let [nxt (decode-step m dstate (peek ids) pos)]
                              (if (or (= nxt 2) (>= (inc pos) max-tokens))
                                ids
                                (recur (conj ids nxt) (inc pos)))))]
                  (assoc state :ids ids :text (decode-tokens m (rest ids))))
                state)]
    state))

(defn transcribe-ts
  "Like transcribe, but returns {:text s :words [{:word :start :end}]} — word-level
  timestamps from monotonic cross-attention alignment (20ms encoder frames)."
  [m wav]
  (let [{:keys [^floats samples]} (if (string? wav) (audio/load-wav wav) wav)
        n (* (quot (alength samples) 80) 80)
        pcm (java.util.Arrays/copyOfRange samples 0 (int n))
        {:keys [x T]} (frontend m pcm)
        enc (encode m x T)
        mem (adapter m enc T 0)
        state (decoder-state m mem T)
        max-tokens (min 256 (long (Math/ceil (* 6.5 (/ n 16000.0)))))
        Tl (long T)
        ;; greedy with attention capture
        [ids rows] (loop [ids [1] rows [] pos 0]
                     (let [w (float-array Tl)
                           nxt (decode-step m state (peek ids) pos w)]
                       (if (or (= nxt 2) (>= (inc pos) max-tokens))
                         [ids (conj rows w)]
                         (recur (conj ids nxt) (conj rows w) (inc pos)))))
        out-ids (vec (rest ids))                            ;; tokens emitted (rows align)
        ;; DTW alignment over the [tokens x frames] attention matrix (greedy argmax
        ;; lets diffuse early rows collapse the start — the whisper lesson)
        nt (count out-ids)
        cost (let [c (make-array Double/TYPE nt Tl)]
               (dotimes [i nt]
                 (let [^floats w (nth rows i)
                       sum (loop [j 0 s 1.0e-9] (if (< j Tl) (recur (inc j) (+ s (aget w j))) s))]
                   (dotimes [j Tl]
                     (aset ^doubles (aget ^objects c i) j (- (/ (aget w j) sum))))))
               c)
        frames (let [D (make-array Double/TYPE (inc nt) (inc Tl))
                     P (make-array Byte/TYPE (inc nt) (inc Tl))]
                 (dotimes [j (inc Tl)] (aset ^doubles (aget ^objects D 0) j
                                             (if (zero? j) 0.0 1.0e18)))
                 (dotimes [i nt] (aset ^doubles (aget ^objects D (inc i)) 0 1.0e18))
                 (dotimes [i nt]
                   (let [^doubles ci (aget ^objects cost i)
                         ^doubles dprev (aget ^objects D i)
                         ^doubles dcur (aget ^objects D (inc i))
                         ^bytes pcur (aget ^objects P (inc i))]
                     (dotimes [j Tl]
                       (let [diag (aget dprev j) up (aget dprev (inc j)) left (aget dcur j)
                             m (min diag (min up left))]
                         (aset dcur (inc j) (+ (aget ci j) m))
                         (aset pcur (inc j) (byte (cond (= m diag) 0 (= m up) 1 :else 2)))))))
                 ;; backtrace: token i -> first frame on its path row
                 (loop [i nt j Tl acc (transient (vec (repeat nt 0)))]
                   (if (pos? i)
                     (let [p (aget ^bytes (aget ^objects P i) j)
                           acc (assoc! acc (dec i) (dec j))]
                       (case (int p)
                         0 (recur (dec i) (dec j) acc)
                         1 (recur (dec i) j acc)
                         2 (recur i (dec j) acc)))
                     (persistent! acc))))
        ;; group SP tokens into words (▁ starts a word)
        words (loop [i 0 cur nil acc []]
                (if (< i (count out-ids))
                  (let [tok (get (:id->tok m) (long (nth out-ids i)) "")
                        starts? (or (.startsWith ^String tok "▁") (nil? cur))
                        piece (str/replace tok "▁" (if (.startsWith ^String tok "▁") " " ""))]
                    (if (and starts? cur)
                      (recur (inc i) {:word piece :start (nth frames i) :end (nth frames i)}
                             (conj acc cur))
                      (recur (inc i)
                             (if cur
                               (-> cur (update :word str piece) (assoc :end (nth frames i)))
                               {:word piece :start (nth frames i) :end (nth frames i)})
                             acc)))
                  (if cur (conj acc cur) acc)))
        fin (fn [w] {:word (str/trim (:word w))
                     :start (* 0.02 (double (:start w)))
                     :end (* 0.02 (inc (double (:end w))))})]
    {:text (decode-tokens m out-ids)
     :words (mapv fin (remove #(str/blank? (:word %)) words))}))

(defn transcribe
  "WAV path (or {:samples :rate} from pretrained.audio) → transcription string."
  [m wav]
  (let [{:keys [^floats samples]} (if (string? wav) (audio/load-wav wav) wav)
        n (* (quot (alength samples) 80) 80)
        pcm (java.util.Arrays/copyOfRange samples 0 (int n))
        {:keys [x T]} (frontend m pcm)
        enc (encode m x T)
        mem (adapter m enc T 0)
        state (decoder-state m mem T)
        max-tokens (min 256 (long (Math/ceil (* 6.5 (/ n 16000.0)))))]
    (loop [ids [1] pos 0]
      (let [nxt (decode-step m state (peek ids) pos)]
        (if (or (= nxt 2) (>= (inc pos) max-tokens))
          (decode-tokens m (rest ids))
          (recur (conj ids nxt) (inc pos)))))))
