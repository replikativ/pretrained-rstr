(ns pretrained.decoder
  "Generic descriptor-driven decoder-LM engine (toward pretrained-rstr).

  A standard decoder architecture is expressed as DATA — a descriptor — and this one
  engine runs it: weight access by ROLE (a per-arch name->template map), hyperparameters
  from the HF config, and a fixed forward skeleton whose variation points are FLAGS:

    pre-norm -> [q,k,v proj (+qk-norm) +rope] -> attn(+sliding-window) -> o-proj
              -> (post-attn-norm) -> residual
              -> ffn-pre-norm -> FFN(gate/up/down, geglu|swiglu) -> (post-ffn-norm) -> residual

  The matmuls run on the raster Q4 :stream-gemv kernel + persistent spin-pool; norms,
  attention and KV stay f32. The blocks (rms-norm, gqa attention, geglu) are raster
  deftms, so the same descriptor can later target the JVM bytecode or GPU backends by
  swapping the compute profile — the descriptor never names a backend.

  A new standard architecture = a descriptor (names + dims + flags), no engine code."
  (:require [pretrained.sampling :as samp]
            [raster.dl.nn :as nn]
            [raster.dl.attention :as attn]
            [raster.par :as par]
            [pretrained.safetensors :as st]
            [raster.quant.op :as ql]
            [raster.compiler.backend.cpu.quant :as cq]
            [clojure.data.json :as json]))

;; ---------------------------------------------------------------------------
;; Generic HF config + weights -> base model map (arch-agnostic)
;; ---------------------------------------------------------------------------

(defn load-hf
  "Read an HF model dir (config.json + model.safetensors) into the standard decoder-LM
  hparam map + weights. Optional/gemma-ish fields default sensibly so the SAME loader
  serves any standard decoder; per-arch specifics live entirely in the descriptor."
  [dir]
  (let [cfg (json/read-str (slurp (str dir "/config.json")) :key-fn keyword)
        head-dim (or (:head_dim cfg) (quot (:hidden_size cfg) (:num_attention_heads cfg)))]
    {:config cfg
     :dir dir
     ;; Normalize the optional "model." prefix away: causal-LM checkpoints store
     ;; "model.layers...", base-model exports (e.g. Qwen3-Embedding) store "layers..."
     ;; bare. Descriptors use the bare names.
     :weights (into {} (map (fn [[k v]]
                              [(if (clojure.string/starts-with? k "model.") (subs k 6) k) v]))
                    (st/load-safetensors (str dir "/model.safetensors")))
     :n-layers (:num_hidden_layers cfg)
     :d-model (:hidden_size cfg)
     :d-ff (:intermediate_size cfg)
     :n-q (:num_attention_heads cfg)
     :n-kv (or (:num_key_value_heads cfg) (:num_attention_heads cfg))
     :head-dim head-dim
     :vocab (:vocab_size cfg)
     :eps (double (or (:rms_norm_eps cfg) 1.0e-6))
     :rope-global (double (or (:rope_theta cfg) 10000.0))
     :rope-local (double (or (:rope_local_base_freq cfg) 10000.0))
     :attn-scale (/ 1.0 (Math/sqrt (double (or (:query_pre_attn_scalar cfg) head-dim))))}))

;; ---------------------------------------------------------------------------
;; Role / name resolution + weight shapes (the data layer)
;; ---------------------------------------------------------------------------

(defn role-name
  "Resolve a weight role to its HF tensor name for layer l (nil l = global weight)."
  [desc role l]
  (let [t (get-in desc [:names role])]
    (assert t (str "no name template for role " role " in " (:arch desc)))
    (if l (format t (int l)) t)))

(defn moe-name
  "Expert weight tensor name: role :gate/:up/:down, layer l, expert e."
  [desc role l e]
  (format (get-in desc [:moe-names role]) (int l) (int e)))

(defn- raw [m role l]
  (:data (get (:weights m) (role-name (:desc m) role l))))

(defn- linear-shape
  "[in out] of a linear role from the model dims (row-major [out,in] weight)."
  [m role]
  (let [{:keys [d-model d-ff n-q n-kv head-dim vocab]} m
        qd (* n-q head-dim) kd (* n-kv head-dim)]
    (case role
      :attn-q [d-model qd] :attn-k [d-model kd] :attn-v [d-model kd]
      :attn-o [qd d-model]
      :ffn-gate [d-model d-ff] :ffn-up [d-model d-ff] :ffn-down [d-ff d-model]
      :embed [d-model vocab] :lm-head [d-model vocab])))

;; ---------------------------------------------------------------------------
;; Quantize all linear weights into the Q4 :stream-gemv layout
;; ---------------------------------------------------------------------------

(defn stream-cache-valid? [^java.io.File f ^java.io.File src]
  (and (.exists f) (.exists src)
       (with-open [in (java.io.DataInputStream. (java.io.BufferedInputStream. (java.io.FileInputStream. f)))]
         (and (= 0x51535452 (.readInt in))                ;; "QSTR"
              (= (.length src) (.readLong in))
              (= (.lastModified src) (.readLong in))))))

(defn- write-arr! [^java.io.DataOutputStream out a]
  (cond
    (instance? (Class/forName "[B") a)
    (do (.writeByte out 0) (.writeInt out (alength ^bytes a)) (.write out ^bytes a))
    (instance? (Class/forName "[I") a)
    (let [^ints a a bb (java.nio.ByteBuffer/allocate (* 4 (alength a)))]
      (.writeByte out 1) (.writeInt out (alength a))
      (.put (.asIntBuffer bb) a) (.write out (.array bb)))
    (instance? (Class/forName "[F") a)
    (let [^floats a a bb (java.nio.ByteBuffer/allocate (* 4 (alength a)))]
      (.writeByte out 2) (.writeInt out (alength a))
      (.put (.asFloatBuffer bb) a) (.write out (.array bb)))
    :else (throw (ex-info (str "unserializable array " (class a)) {}))))

(defn- read-arr [^java.io.DataInputStream in]
  (let [tag (.readByte in) n (.readInt in)]
    (case (int tag)
      0 (let [b (byte-array n)] (.readFully in b) b)
      1 (let [b (byte-array (* 4 n)) out (int-array n)]
          (.readFully in b)
          (.get (.asIntBuffer (java.nio.ByteBuffer/wrap b)) out) out)
      2 (let [b (byte-array (* 4 n)) out (float-array n)]
          (.readFully in b)
          (.get (.asFloatBuffer (java.nio.ByteBuffer/wrap b)) out) out))))

(defn- write-stream-cache!
  "Serialize quantized streams: header (magic, src size+mtime) + per-entry
  name/in/out + type-tagged wqi/wsi. One-time cost after first quantize; later
  loads skip the ~30s quantization entirely."
  [^java.io.File f ^java.io.File src w]
  (let [tmp (java.io.File. (str (.getPath f) ".tmp"))]
    (with-open [out (java.io.DataOutputStream. (java.io.BufferedOutputStream. (java.io.FileOutputStream. tmp) (* 1024 1024)))]
      (.writeInt out 0x51535452)
      (.writeLong out (.length src))
      (.writeLong out (.lastModified src))
      (.writeInt out (count w))
      (doseq [[nm e] w]
        (.writeUTF out nm)
        (.writeInt out (int (:in e))) (.writeInt out (int (:out e)))
        (write-arr! out (:wqi e))
        (write-arr! out (:wsi e))))
    (.renameTo tmp f)))

(defn- read-stream-cache [^java.io.File f]
  (with-open [in (java.io.DataInputStream. (java.io.BufferedInputStream. (java.io.FileInputStream. f) (* 1024 1024)))]
    (.readInt in) (.readLong in) (.readLong in)
    (let [n (.readInt in)]
      (into {} (for [_ (range n)]
                 (let [nm (.readUTF in)
                       in* (.readInt in) out* (.readInt in)
                       wqi (read-arr in) wsi (read-arr in)]
                   [nm {:wqi wqi :wsi wsi :in (long in*) :out (long out*)}]))))))

(defn quantize-stream
  "Repack every :linear-role weight (per layer + the globals) into the stream-gemv
  layout. Returns {:fmt fmt :w {hf-name {:wqi :wsi :in :out}}}. `fmt` :q4 (decode
  default) or :q8 (embedder quality — 4-bit weights cost ~5% embedding cosine,
  Q8_0 is lossless; measured on Qwen3-Embedding-0.6B)."
  ([m] (quantize-stream m :q4))
  ([m fmt]
  (let [desc (:desc m)
        format (case fmt :q4 cq/q4-0 :q8 cq/q8-0)
        per-layer (:linear-roles desc)
        moe (get-in desc [:flags :moe])
        names+shapes (concat
                      (for [role (:global-linear-roles desc)]
                        [(role-name desc role nil) (linear-shape m role)])
                      (for [l (range (:n-layers m)) role per-layer]
                        [(role-name desc role l) (linear-shape m role)])
                      ;; MoE: every expert's SwiGLU (gate/up: [d,inter], down: [inter,d])
                      (when (map? moe)
                        (for [l (range (:n-layers m)) e (range (:experts moe))
                              [role sh] [[:gate [(:d-model m) (:inter moe)]]
                                         [:up [(:d-model m) (:inter moe)]]
                                         [:down [(:inter moe) (:d-model m)]]]]
                          [(moe-name desc role l e) sh])))]
    {:fmt fmt
     ;; matmuls run through the raster.quant.op/qlinear-i8[-q8] compiler op (it owns
     ;; the host-best int8-MAC kernel). Quantized streams are DISK-CACHED next to
     ;; the weights (keyed by safetensors size+mtime): first load pays the ~30s
     ;; quantize once, later loads read back in ~2s.
     :w (let [src (when (:dir m) (java.io.File. (str (:dir m) "/model.safetensors")))
              cf (when src (java.io.File. (str (:dir m) "/.qstream-" (name fmt) ".bin")))]
          (if (and cf (stream-cache-valid? cf src))
            (read-stream-cache cf)
            (let [w (into {} (pmap (fn [[nm [in out]]]
                                     (let [{:keys [wq ws]} (cq/quantize-weight (:data (get (:weights m) nm)) format)]
                                       [nm (assoc (cq/repack-stream wq ws out in format) :in in :out out)]))
                                   names+shapes))]
              (when cf (write-stream-cache! cf src w))
              w)))})))

;; ---------------------------------------------------------------------------
;; Forward primitives (descriptor-aware)
;; ---------------------------------------------------------------------------

(defn- embed
  "Token -> scaled embedding row [d_model]."
  [m token]
  (let [{:keys [d-model]} m
        emb (raw m :embed nil)
        scale (if (= :sqrt-d (get-in m [:desc :flags :embed-scale])) (Math/sqrt (double d-model)) 1.0)
        x (float-array d-model) base (* (long token) d-model)]
    (dotimes [j d-model]
      (aset x j (float (* scale (aget ^floats emb (+ base j))))))
    x))

(defn- qsl
  "Quantized stream linear for weight `nm`: the raster.quant.op/qlinear-i8 compiler
  op (int8-quantizes the activation + runs the host int8-MAC kernel)."
  ^floats [qm nm ^floats x]
  (let [{:keys [wqi wsi in out]} (get (:w qm) nm)]
    (if (= :q8 (:fmt qm))
      (ql/qlinear-i8-q8 x wqi wsi in out)
      (ql/qlinear-i8 x wqi wsi in out))))

(defn- global-layer?
  "Is layer l a full/global-attention layer? From descriptor data: an explicit set
  :global-layers, else a :global-layer-pattern p (every p-th, gemma-style), else all."
  [m l]
  (let [flags (get-in m [:desc :flags])]
    (cond (:global-layers flags) (contains? (:global-layers flags) l)
          (:global-layer-pattern flags) (zero? (mod (inc l) (long (:global-layer-pattern flags))))
          :else true)))

(defn- rope-theta [m l]
  (if (= :dual (get-in m [:desc :flags :rope]))
    (if (global-layer? m l) (:rope-global m) (:rope-local m))
    (:rope-global m)))

;; ---------------------------------------------------------------------------
;; The generic decode step
;; ---------------------------------------------------------------------------

(defn- tapped
  "Apply a tap at a named point. nil tap → arr unchanged (the hot path: one predicted
  branch, no work). A tap returning nil is a READ (keep arr); returning an array is a
  WRITE (use it). This single seam is read + intervene."
  ^floats [tap point l ^floats arr]
  (if tap (let [r (tap point l arr)] (if (nil? r) arr r)) arr))

(defn decode-step
  "One token forward at absolute position p over the descriptor. Matmuls on the Q4
  stream pool; norms/attention/KV f32.

  Optional `tap` (a fn `[point layer ^floats arr] -> arr'|nil`) is the UNIFIED
  read/write seam: it is called at each canonical hook point with that activation.
  Returning nil = a pure READ (the activation flows on unchanged — snapshot it to
  cache it). Returning an array = a WRITE/intervention (that array replaces the
  activation going forward — steering, ablation, patching, a trainable edit). Points
  fired per layer: :resid-pre (block input), :attn-out (post o-proj), :resid-mid
  (post-attn residual), :mlp-out (ffn/moe output), :resid-post (block output). It is
  nil on the default 5-arg path, so the hot forward pays NOTHING: each point is a
  single predicted branch (see `tapped`), no snapshot, no allocation. The tap gets the
  live array — copy it if you retain it past the call."
  ([m token p kc vc] (decode-step m token p kc vc nil))
  ([m token p kc vc tap]
  (let [{:keys [d-model n-layers n-q n-kv head-dim eps attn-scale qm desc]} m
        {:keys [flags]} desc
        go (double (get-in flags [:norm :gain-offset] 0.0))
        qk? (:qk-norm flags)
        sand? (:sandwich-norms flags)
        ffn-geglu? (= :geglu (:ffn flags))
        sw (:sliding-window flags)
        kvrow (* n-kv head-dim)
        nm (fn [role l] (role-name desc role l))
        ;; token may be a pre-computed embedding row (float[]) — multimodal splice
        ;; (Qwen3-ASR feeds projector rows at <|audio_pad|> positions)
        xN (loop [x (if (number? token) (embed m token) ^floats token) l 0]
             (if (< l n-layers)
               (let [x (tapped tap :resid-pre l x)
                     theta (rope-theta m l)
                     h (nn/rms-norm x (raw m :attn-norm l) 1 d-model eps go)
                     q (cond-> (qsl qm (nm :attn-q l) h)
                         qk? (nn/rms-norm (raw m :attn-q-norm l) n-q head-dim eps go)
                         true (attn/rope-pos 1 n-q head-dim theta p))
                     knew (cond-> (qsl qm (nm :attn-k l) h)
                            qk? (nn/rms-norm (raw m :attn-k-norm l) n-kv head-dim eps go)
                            true (attn/rope-pos 1 n-kv head-dim theta p))
                     vnew (qsl qm (nm :attn-v l) h)
                     ^floats kcl (nth kc l) ^floats vcl (nth vc l)
                     _ (System/arraycopy knew 0 kcl (* p kvrow) kvrow)
                     _ (System/arraycopy vnew 0 vcl (* p kvrow) kvrow)
                     cl (inc p) group (quot n-q n-kv)
                     kv-start (if (and sw (not (global-layer? m l)))
                                (max 0 (- cl (long (:size sw)))) 0)
                     ao (float-array (* n-q head-dim))
                     _ (cq/run-par-fn!
                        (fn [wid n]
                          (let [chunk (quot (+ n-q (dec (int n))) (int n))
                                h0 (* (int wid) chunk) hc (min chunk (- n-q h0))]
                            (when (pos? hc)
                              (attn/gqa-decode-attention-heads! q kcl vcl ao cl kv-start h0 hc group n-kv head-dim attn-scale)))))
                     o (qsl qm (nm :attn-o l) ao)
                     o (if sand? (nn/rms-norm o (raw m :attn-post-norm l) 1 d-model eps go) o)
                     o (tapped tap :attn-out l o)
                     x1 (nn/residual-add x o d-model)
                     x1 (tapped tap :resid-mid l x1)
                     f (nn/rms-norm x1 (raw m :ffn-pre-norm l) 1 d-model eps go)
                     dn (if-let [moe (and (map? (get flags :moe)) (get flags :moe))]
                          ;; MoE: router f32 -> softmax over ALL experts -> top-k of the
                          ;; PROBABILITIES -> renorm -> weighted sum of expert SwiGLUs
                          (let [E (long (:experts moe)) K (long (:top-k moe)) inter (long (:inter moe))
                                ^floats gw (raw m :moe-router l)
                                logits (nn/linear-nb f gw 1 (long d-model) E)
                                ^floats probs (nn/softmax-1d logits E)
                                top (vec (take K (sort-by #(- (aget probs (int %))) (range E))))
                                psum (if (:norm-topk? moe)
                                       (reduce (fn [s e] (+ s (double (aget probs (int e))))) 0.0 top)
                                       1.0)]
                            (reduce (fn [acc e]
                                      (let [w (/ (double (aget probs (int e))) psum)
                                            g (nn/silu (qsl qm (moe-name desc :gate l e) f) inter)
                                            u (qsl qm (moe-name desc :up l e) f)
                                            gu (nn/hadamard g u inter)
                                            d (qsl qm (moe-name desc :down l e) gu)]
                                        (par/axpy w d acc)))
                                    (float-array (long d-model)) top))
                          (let [g (qsl qm (nm :ffn-gate l) f)
                                g (if ffn-geglu? (nn/gelu g (:d-ff m)) (nn/silu g (:d-ff m)))
                                u (qsl qm (nm :ffn-up l) f)
                                gu (nn/hadamard g u (:d-ff m))]
                            (qsl qm (nm :ffn-down l) gu)))
                     dn (if sand? (nn/rms-norm dn (raw m :ffn-post-norm l) 1 d-model eps go) dn)
                     dn (tapped tap :mlp-out l dn)
                     x' (nn/residual-add x1 dn d-model)
                     x' (tapped tap :resid-post l x')]
                 (recur x' (inc l)))
               x))]
    (nn/rms-norm xN (raw m :final-norm nil) 1 d-model eps go))))

(defn hidden-states
  "Run `ids` through the KV-cached forward and capture the per-layer residual stream
  (resid-post) at the LAST position, plus the final-normed hidden. Returns
  `{:layers [float[d] × n-layers], :final float[d]}`. The capturing tap fires ONLY on
  the last position; every earlier position takes the zero-cost default path — so this
  is opt-in latent capture with no change to the hot forward. Feed `:final` (or any
  `:layers` entry) to `lm-logits` for a logit lens; stack `:layers` for probing."
  [m ids]
  (let [{:keys [n-layers n-kv head-dim]} m
        ids (vec ids) P (count ids)
        slot (* P (long n-kv) (long head-dim))
        kc (vec (repeatedly n-layers #(float-array slot)))
        vc (vec (repeatedly n-layers #(float-array slot)))
        cap (object-array n-layers)
        tap (fn [_point l ^floats arr] (aset cap (int l) (aclone arr)))]
    (loop [p 0 h nil]
      (if (< p P)
        (recur (inc p) (decode-step m (nth ids p) p kc vc (when (= p (dec P)) tap)))
        {:layers (vec cap) :final h}))))

(defn lm-logits
  "Logits over the vocab for a final-normed hidden state (tied embed or untied lm_head)."
  ^floats [m hidden]
  (let [role (if (get-in m [:desc :flags :tied-lm-head] true) :embed :lm-head)]
    (qsl (:qm m) (role-name (:desc m) role nil) hidden)))

(defn logit-lens
  "Vocab logits for a RAW layer resid-post (e.g. a `hidden-states` :layers entry):
  apply the final RMS-norm, then the (tied) lm head — the logit lens (nostalgebraist
  2020), a per-layer next-token readout. Use to watch the prediction sharpen with
  depth; argmax each layer's logits and decode to see the token trajectory."
  ^floats [m resid]
  (let [{:keys [d-model eps desc]} m
        go (double (get-in desc [:flags :norm :gain-offset] 0.0))]
    (lm-logits m (nn/rms-norm resid (raw m :final-norm nil) 1 d-model eps go))))

(def ^:private all-tap-points #{:resid-pre :attn-out :resid-mid :mlp-out :resid-post})

(defn trace
  "The unified read/write inspection seam (run_with_cache + interventions). Runs `ids`
  through the KV-cached forward and returns `{:logits float[vocab] (last position),
  :cache {[point layer] float[]}}`.

  opts:
    :capture  a set of tap points to record (or :all), snapshotted at the LAST
              position — the read side (probing, logit-lens, SAE data).
    :edit     a fn `[point layer ^floats arr] -> arr'|nil` applied at EVERY position —
              the write side. Return a modified array to intervene (steering =
              arr+v at :resid-post, ablation = zeros at :mlp-out/:attn-out, activation
              patching = a captured clean activation), nil to leave it unchanged. On
              GPU the same edit is an injected deftm (see decoder-gpu); here it is an
              eager callback. Interventions must be applied consistently across
              positions to be causal.

  With neither :capture nor :edit this is just `generate`-style forward returning logits."
  [m ids & {:keys [capture edit]}]
  (let [{:keys [n-layers n-kv head-dim]} m
        ids (vec ids) P (count ids)
        slot (* P (long n-kv) (long head-dim))
        kc (vec (repeatedly n-layers #(float-array slot)))
        vc (vec (repeatedly n-layers #(float-array slot)))
        points (cond (= capture :all) all-tap-points (set? capture) capture
                     (nil? capture) #{} :else (set capture))
        cache (atom {})
        mk-tap (fn [p]
                 (when (or edit (and (= p (dec P)) (seq points)))
                   (fn [point l ^floats arr]
                     (let [r (when edit (edit point l arr))
                           v (if (nil? r) arr r)]
                       (when (and (= p (dec P)) (contains? points point))
                         (swap! cache assoc [point l] (aclone ^floats v)))
                       r))))
        h (loop [p 0 h nil]
            (if (< p P) (recur (inc p) (decode-step m (nth ids p) p kc vc (mk-tap p))) h))]
    {:logits (lm-logits m h) :cache @cache}))

(defn- hidden->token [m hidden sampler]
  (sampler (lm-logits m hidden) (:vocab m)))

(defn embed-prefill
  "Prefill `ids` through the KV-cached forward and return the LAST token's hidden state
  (after final norm), L2-normalized — last-token pooling, the decoder-embedder convention
  (Qwen3-Embedding). Returns float[d-model]. Tokenization conventions (instruction
  prefix, EOS append) are the caller's responsibility."
  ^floats [m ids]
  (let [{:keys [n-layers n-kv head-dim]} m
        ids (vec ids) P (count ids)
        slot (* P (long n-kv) (long head-dim))
        kc (vec (repeatedly n-layers #(float-array slot)))
        vc (vec (repeatedly n-layers #(float-array slot)))
        ^floats h (loop [p 0 h nil]
                    (if (< p P) (recur (inc p) (decode-step m (nth ids p) p kc vc)) h))]
    (nn/l2-normalize! h (long (alength h)))))

(defn generate-cached
  "KV-cache generation; returns the new token ids. `sampler` is a fn [logits vocab] ->
  token-id (default greedy argmax — see pretrained.sampling/make-sampler)."
  ([m prompt-ids n] (generate-cached m prompt-ids n samp/greedy))
  ([m prompt-ids n sampler]
   (let [{:keys [n-layers n-kv head-dim]} m
         prompt (vec prompt-ids) P (count prompt)
         slot (* (+ P n) n-kv head-dim)
         kc (vec (repeatedly n-layers #(float-array slot)))
         vc (vec (repeatedly n-layers #(float-array slot)))
         h0 (loop [p 0 h nil] (if (< p P) (recur (inc p) (decode-step m (nth prompt p) p kc vc)) h))]
     (loop [h h0 pos P out []]
       (if (< (count out) n)
         (let [t (hidden->token m h sampler)]
           (recur (decode-step m t pos kc vc) (inc pos) (conj out t)))
         out)))))
