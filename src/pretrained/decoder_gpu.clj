(ns pretrained.decoder-gpu
  "GPU-resident decode — the device twin of pretrained.decoder/decode-step.

  Same descriptor + model map; instead of the CPU spin-pool, the whole decoder layer is
  compiled ONCE into a Level-Zero command graph (compile-gpu-program) over resident buffers
  (weights :constant, KV :state) and replayed per token. Device-side position (posbuf/clenbuf)
  so the graph binds once and the host only writes the token + replays.

  The decoder layer + head deftms are GENERATED from the model's descriptor flags
  (gen-layer!/gen-head!) — the hand-written gemma-3-270m layer this generator was
  validated against lives on as the frozen oracle fixture in
  pretrained.decoder-gpu-oracle-test.

  Matmuls use the GPU Q4K dp4a kernels (raster.quant.kernels-k); the weight packers
  for the kernel layouts live next to the kernels (raster.quant.pack) — this ns only
  walks the descriptor's linear roles and packs one tensor at a time."
  (:require [pretrained.decoder :as dec]
            [raster.arrays :as arr]
            [raster.quant.kernels-k :as qk]
            [raster.quant.pack :as qpack]
            [raster.dl.nn :as nn]
            [raster.dl.attention :as attn]
            [raster.gpu.core :as gpu]
            [raster.compiler.pipeline :as pipeline]))

;; ---------------------------------------------------------------------------
;; Q4K weight quantization (GPU dp4a layout; packer = raster.quant.pack/q4k-of)
;; ---------------------------------------------------------------------------

(defn gpu-quantize
  "Q4K-quantize every linear-role weight (per-layer + globals) from the model map `m`.
  Returns {hf-name -> {:wp :da :db :aq :bq :in :out}}. EAGER and memory-bounded:
  strictly one tensor at a time, so peak heap = the source f32 weight (+ its padded
  copy for the tied vocab×d_model lm-head) + one packed result — NOT one per pmap
  chunk (the previous chunked-lazy pmap realized up to 32 quantize jobs at once and
  drove multi-GB heap spikes at bind time)."
  [m]
  (let [desc (:desc m)
        names+shapes (vec (concat
                           (for [role (:global-linear-roles desc)]
                             [(dec/role-name desc role nil) (#'dec/linear-shape m role)])
                           (for [l (range (:n-layers m)) role (:linear-roles desc)]
                             [(dec/role-name desc role l) (#'dec/linear-shape m role)])))]
    (persistent!
     (reduce (fn [acc [nm [in out]]]
               (assoc! acc nm (qpack/q4k-of (:data (get (:weights m) nm)) out in)))
             (transient {})
             names+shapes))))

;; The hand-written gemma-3-270m layer/head deftms (gemma-layer!/gemma-layer-fn!/
;; gemma-head!, ported from the validated /tmp gcl2!/gcomp-head!, oracle 236761)
;; were dead in production — bind-decode! only ever uses the generated gen-layer!/
;; gen-head! — and now live as the frozen oracle fixture in
;; pretrained.decoder-gpu-oracle-test, which enforces that the generated layer
;; reproduces them bit-exactly.

;; ---------------------------------------------------------------------------
;; On-device decode tail: argmax(logits) → tokbuf, embedding gather → next r0.
;; Appended to the resident graph so a replay produces the NEXT token's residual input on-device —
;; the host uploads only the 16-byte position and downloads the 4-byte token id (no 1MB logits
;; download, no embedding upload). The embedding table is bound PRE-SCALED by sqrt(d) (double-math
;; host-side, bit-identical to embed-row) so the gather is a pure copy. Argmax ties are resolved
;; by whichever work-item writes last (CPU argmax picks the first index) — irrelevant for logits.
;; ---------------------------------------------------------------------------

(raster.core/deftm decode-tail!
  [logits :- (Array float) emb :- (Array float) tokbuf :- (Array int) r0 :- (Array float)
   vocab :- Long d :- Long] :- Void
  (let [mx (raster.par/reduce acc -1.0e30 i vocab
             (raster.numeric/max acc (raster.arrays/aget logits i)))]
    (raster.par/map-void! j vocab
      (when (raster.numeric/== (raster.arrays/aget logits j) mx)
        (raster.arrays/aset tokbuf 0 j)))
    (raster.par/map-void! j d
      (raster.arrays/aset r0 j
        (raster.arrays/aget emb (+ (* (raster.arrays/aget tokbuf 0) d) j))))))

(defn- embed-scale ^double [m]
  (if (= :sqrt-d (get-in m [:desc :flags :embed-scale]))
    (Math/sqrt (double (:d-model m))) 1.0))

(defn- prescaled-embed
  "The embedding table pre-scaled by the descriptor's embed-scale (sqrt-d or 1), double-math per element (matches embed-row)."
  ^floats [m]
  (let [^floats emb (:data (get (:weights m) (dec/role-name (:desc m) :embed nil)))
        n (alength emb) scale (embed-scale m)
        out (float-array n)]
    (dotimes [i n] (aset out i (float (* scale (aget emb i)))))
    out))

;; ---------------------------------------------------------------------------
;; Descriptor-GENERATED layer + head. The layer deftm is emitted from the
;; model's descriptor flags (qk-norm / sandwich-norms / geglu|swiglu) + dims and
;; eval'd once per (arch, dims, rms-style) — Julia-style per-config specialization
;; (dims as literals, so kernels constant-fold). Oracle: the gemma-3-270m generated
;; layer/head reproduce the validated hand-written fixture bit-exactly — enforced
;; by pretrained.decoder-gpu-oracle-test.
;; Distinct names per config avoid the deftm stale-arity reload trap.
;; ---------------------------------------------------------------------------

(defn- dims-of
  "Derived GPU-layer dimensions (Q4K needs matmul IN-dims padded to 256)."
  [m]
  (let [d (long (:d-model m)) dff (long (:d-ff m))
        nq (long (:n-q m)) nkv (long (:n-kv m)) hd (long (:head-dim m))
        qd (* nq hd) kvrow (* nkv hd)
        dp (qpack/nextpad d) qdp (qpack/nextpad qd) dffp (qpack/nextpad dff)]
    {:d d :dff dff :nq nq :nkv nkv :hd hd :qd qd :kvrow kvrow
     :dp dp :qdp qdp :dffp dffp
     :nsb-d (quot dp 256) :nsb-qd (quot qdp 256) :nsb-dff (quot dffp 256)
     :group (quot nq nkv)}))

(def ^:private gen-cache (atom {}))
(def af {:f '(Array float) :i '(Array int) :b '(Array byte) :l '(Array long)})

(defn- wp5 [pl]
  (mapcat (fn [[suf t]] [(symbol (str "w" pl suf)) :- (af t)])
          [["p" :i] ["da" :f] ["db" :f] ["aq" :b] ["bq" :b]]))

(defn- layer-form
  "The GPU layer deftm form for model m: flags decide qk-norm / sandwich / gelu|silu
  forms (and their params); dims are baked as literals. rms-style :fn = Stage-A
  parallel-reduce rms (fast); :map-void = serial rms (the oracle anchor)."
  [m nm rms-style steer?]
  (let [{:keys [d dp dff dffp qd qdp kvrow nq nkv hd group nsb-d nsb-qd nsb-dff]} (dims-of m)
        flags (get-in m [:desc :flags])
        qk? (boolean (:qk-norm flags))
        sand? (boolean (:sandwich-norms flags))
        act-mul (if (= :geglu (:ffn flags)) 'nn/gelu-mul! 'nn/silu-mul!)
        go (double (get-in flags [:norm :gain-offset] 0.0))
        rms1 (fn [x w out]
               (if (= rms-style :fn)
                 (list 'nn/rms-norm-1row! x w out d 'eps go)
                 (list 'nn/rms-norm! x w out 1 d 'eps go)))
        params (vec (concat
                     ['r-in :- (af :f) 'inln :- (af :f)]
                     (when qk? ['qln :- (af :f) 'kln :- (af :f)])
                     (when sand? ['paln :- (af :f)])
                     ['pfln :- (af :f)]
                     (when sand? ['pffln :- (af :f)])
                     (mapcat wp5 ["q" "k" "v" "o" "g" "u" "d"])
                     ['kc :- (af :f) 'vc :- (af :f)
                      'h :- (af :f) 'qinp :- (af :i) 'qins :- (af :f) 'qinb :- (af :i)
                      'q :- (af :f) 'k :- (af :f) 'v :- (af :f)]
                     (when qk? ['qn :- (af :f) 'kn :- (af :f)])
                     ['qr :- (af :f) 'kr :- (af :f) 'sc :- (af :f) 'at :- (af :f)
                      'qap :- (af :i) 'qas :- (af :f) 'qab :- (af :i)
                      'o :- (af :f)]
                     (when sand? ['o2 :- (af :f)])
                     ['xmid :- (af :f) 'f :- (af :f)
                      'qfp :- (af :i) 'qfs :- (af :f) 'qfb :- (af :i)
                      'gate :- (af :f) 'up :- (af :f) 'hh :- (af :f)
                      'qhp :- (af :i) 'qhs :- (af :f) 'qhb :- (af :i)
                      'down :- (af :f)]
                     (when sand? ['down2 :- (af :f)])
                     ['r-out :- (af :f)]
                     ;; Phase-2b intervention: a resident per-layer steering vector added to
                     ;; resid-post (activation addition). Injected ONLY when steer? — empty →
                     ;; identical program (compile-time switch, fast path untouched).
                     (when steer? ['steer :- (af :f)])
                     ['posbuf :- (af :l) 'clenbuf :- (af :l) 'submax :- (af :f)
                      'maxpos :- 'Long 'eps :- 'Double 'theta :- 'Double 'scale :- 'Double]))
        body (remove nil?
              [(rms1 'r-in 'inln 'h)
               (list 'qk/quant-act-q8k-gpu! 'h 'qinp 'qins 'qinb 'submax nsb-d)
               (list 'qk/qmatmul-q4k-dp4a! 'qinp 'qins 'qinb 'wqp 'wqda 'wqdb 'wqaq 'wqbq 'q dp qd)
               (list 'qk/qmatmul-q4k-dp4a! 'qinp 'qins 'qinb 'wkp 'wkda 'wkdb 'wkaq 'wkbq 'k dp kvrow)
               (list 'qk/qmatmul-q4k-dp4a! 'qinp 'qins 'qinb 'wvp 'wvda 'wvdb 'wvaq 'wvbq 'v dp kvrow)
               (when qk? (list 'nn/rms-norm! 'q 'qln 'qn nq hd 'eps go))
               (when qk? (if (and (= rms-style :fn) (= nkv 1))
                           (list 'nn/rms-norm-1row! 'k 'kln 'kn hd 'eps go)
                           (list 'nn/rms-norm! 'k 'kln 'kn nkv hd 'eps go)))
               (list 'attn/rope-pos-buf! (if qk? 'qn 'q) 'qr nq hd 'theta 'posbuf)
               (list 'attn/rope-pos-buf! (if qk? 'kn 'k) 'kr nkv hd 'theta 'posbuf)
               (list 'attn/kv-append-buf! 'kr 'kc kvrow 'posbuf)
               (list 'attn/kv-append-buf! 'v 'vc kvrow 'posbuf)
               ;; concrete-float layer → cast the Double `scale` to float at the call
               ;; boundary (gqa's scale is :- T; float×double would not lower to GPU C).
               (list 'attn/gqa-decode-attention-buf! 'qr 'kc 'vc 'at 'sc 'clenbuf nq group nkv hd 'maxpos (list 'float 'scale))
               (list 'qk/quant-act-q8k-gpu! 'at 'qap 'qas 'qab 'submax nsb-qd)
               (list 'qk/qmatmul-q4k-dp4a! 'qap 'qas 'qab 'wop 'woda 'wodb 'woaq 'wobq 'o qdp d)
               (when sand? (rms1 'o 'paln 'o2))
               (list 'nn/residual-add! 'r-in (if sand? 'o2 'o) 'xmid d)
               (rms1 'xmid 'pfln 'f)
               (list 'qk/quant-act-q8k-gpu! 'f 'qfp 'qfs 'qfb 'submax nsb-d)
               (list 'qk/qmatmul-q4k-dp4a! 'qfp 'qfs 'qfb 'wgp 'wgda 'wgdb 'wgaq 'wgbq 'gate dp dff)
               (list 'qk/qmatmul-q4k-dp4a! 'qfp 'qfs 'qfb 'wup 'wuda 'wudb 'wuaq 'wubq 'up dp dff)
               (list act-mul 'gate 'up 'hh dff)
               (list 'qk/quant-act-q8k-gpu! 'hh 'qhp 'qhs 'qhb 'submax nsb-dff)
               (list 'qk/qmatmul-q4k-dp4a! 'qhp 'qhs 'qhb 'wdp 'wdda 'wddb 'wdaq 'wdbq 'down dffp d)
               (when sand? (rms1 'down 'pffln 'down2))
               (list 'nn/residual-add! 'xmid (if sand? 'down2 'down) 'r-out d)
               ;; inject the steering add at :resid-post (reuses residual-add!, a known-good
               ;; resident kernel — the tap is "inject a deftm, compile transparently").
               (when steer? (list 'nn/residual-add! 'r-out 'steer 'r-out d))])]
    (list 'raster.core/deftm nm params :- 'Void (cons 'do body))))

(defn- head-form [m nm]
  (let [{:keys [d dp nsb-d]} (dims-of m)
        go (double (get-in m [:desc :flags :norm :gain-offset] 0.0))
        vocab (long (:vocab m))]
    (list 'raster.core/deftm nm
          (vec (concat ['r-fin :- (af :f) 'finalln :- (af :f) 'fh :- (af :f)
                        'hqp :- (af :i) 'hqs :- (af :f) 'hqb :- (af :i)
                        'lmp :- (af :i) 'lmda :- (af :f) 'lmdb :- (af :f)
                        'lmaq :- (af :b) 'lmbq :- (af :b)
                        'logits :- (af :f) 'submax :- (af :f) 'eps :- 'Double]))
          :- 'Void
          (list 'do
                (list 'nn/rms-norm! 'r-fin 'finalln 'fh 1 d 'eps go)
                (list 'qk/quant-act-q8k-gpu! 'fh 'hqp 'hqs 'hqb 'submax nsb-d)
                (list 'qk/qmatmul-q4k-dp4a! 'hqp 'hqs 'hqb 'lmp 'lmda 'lmdb 'lmaq 'lmbq 'logits dp vocab)))))

(defn eval-gen! [nm form]
  (or (get @gen-cache nm)
      (do (binding [*ns* (the-ns 'pretrained.decoder-gpu)] (eval form))
          (let [v (ns-resolve 'pretrained.decoder-gpu nm)]
            (swap! gen-cache assoc nm v)
            v))))

(defn gen-layer!
  "Get-or-create the descriptor-generated GPU layer deftm var for model m."
  [m & {:keys [rms-style steer?] :or {rms-style :fn}}]
  (let [{:keys [d dff]} (dims-of m)
        nm (symbol (str "glayer-" (name (get-in m [:desc :arch])) "-" d "x" dff
                        "-" (name rms-style) (when steer? "-steer") "!"))]
    (eval-gen! nm (layer-form m nm rms-style (boolean steer?)))))

(defn gen-head!
  "Get-or-create the descriptor-generated GPU head deftm var for model m."
  [m]
  (let [{:keys [d]} (dims-of m)
        nm (symbol (str "ghead-" (name (get-in m [:desc :arch])) "-" d "!"))]
    (eval-gen! nm (head-form m nm))))

;; ---------------------------------------------------------------------------
;; Resident multi-layer binding + decode loop
;; One layer program bound once PER LAYER (different weights/KV/residual via pbk) + the head,
;; all into ONE command graph. Scratch buffers are SHARED across layers; weights/norms/KV and the
;; residual stream r{l} are per-layer. PHASE 1 buffer sizes are gemma-3-270m specific.
;; ---------------------------------------------------------------------------

(defn- scratch-dims
  "key -> [size dtype], derived from the model dims (was a static gemma-3-270m map;
  verified identical for 270m). :sc is sized in buffer-specs (needs maxpos)."
  [m]
  (let [{:keys [d dp dff dffp qd qdp kvrow nsb-d nsb-qd nsb-dff]} (dims-of m)
        mx-nsb (max nsb-d nsb-qd nsb-dff)]
    {:h [dp :float] :q [qd :float] :k [kvrow :float] :v [kvrow :float]
     :qn [qd :float] :kn [kvrow :float] :qr [qd :float] :kr [kvrow :float]
     :at [qdp :float] :o [d :float] :o2 [d :float] :xmid [d :float]
     :f [dp :float] :gate [dff :float] :up [dff :float] :hh [dffp :float]
     :down [d :float] :down2 [d :float] :fh [dp :float]
     :qinp [(quot dp 4) :int] :qins [nsb-d :float] :qinb [(quot dp 32) :int]
     :qap [(quot qdp 4) :int] :qas [nsb-qd :float] :qab [(quot qdp 32) :int]
     :qfp [(quot dp 4) :int] :qfs [nsb-d :float] :qfb [(quot dp 32) :int]
     :qhp [(quot dffp 4) :int] :qhs [nsb-dff :float] :qhb [(quot dffp 32) :int]
     :submax [(* 8 mx-nsb) :float]}))

(def ^:private proj->role {"q" :attn-q "k" :attn-k "v" :attn-v "o" :attn-o
                           "g" :ffn-gate "u" :ffn-up "d" :ffn-down})

(defn- norm-roles
  "param-name -> weight role for the per-layer norms this model's flags use."
  [m]
  (let [flags (get-in m [:desc :flags])]
    (cond-> {"inln" :attn-norm "pfln" :ffn-pre-norm}
      (:qk-norm flags)        (assoc "qln" :attn-q-norm "kln" :attn-k-norm)
      (:sandwich-norms flags) (assoc "paln" :attn-post-norm "pffln" :ffn-post-norm))))

(defn- pbk
  "Layer-l buffer key for a layer-program arg name: residual r{l}/r{l+1}, KV kc{l}/vc{l}, weights
  + norms L{l}…, everything else (scratch, posbuf/clenbuf) SHARED by its bare name."
  [norm-names l pn]
  (cond (= pn "r-in")  (keyword (str "r" l))
        (= pn "r-out") (keyword (str "r" (inc l)))
        (= pn "steer") (keyword (str "L" l "steer"))   ; per-layer steering vector (Phase-2b)
        (= pn "kc")    (keyword (str "kc" l))
        (= pn "vc")    (keyword (str "vc" l))
        (norm-names pn) (keyword (str "L" l pn))
        (= \w (first pn))    (keyword (str "L" l pn))
        ;; reduce-result buffers (Stage A's ss_*__rbuf*) must be PER-LAYER, not shared: the
        ;; materialized rms-map reads ss__rbuf while the next layer's reduce overwrites it — a
        ;; cross-layer write-after-read hazard that corrupts results past ~6 layers. 1-elem each.
        (re-find #"__rbuf" pn) (keyword (str "L" l "_" pn))
        :else (keyword pn)))

(defn- head-pbk
  "Head-program arg name -> buffer key. Head reuses the layer's qinp/qins/qinb quant scratch and
  reads the final residual r{n-layers}."
  [n-layers pn]
  (case pn
    "r-fin" (keyword (str "r" n-layers))
    "fh" :fh "hqp" :qinp "hqs" :qins "hqb" :qinb
    (keyword pn)))

(defn- scratch-specs [m]
  (into {} (map (fn [[k [sz ty]]] [k [ty sz nil :scratch]]) (scratch-dims m))))

(defn- weight-specs [m qw l]
  (into {} (mapcat (fn [[pl role]]
                     (let [{:keys [wp da db aq bq]} (get qw (dec/role-name (:desc m) role l))]
                       [[(keyword (str "L" l "w" pl "p"))  [:int   (alength ^ints wp)    wp :constant]]
                        [(keyword (str "L" l "w" pl "da")) [:float (alength ^floats da)  da :constant]]
                        [(keyword (str "L" l "w" pl "db")) [:float (alength ^floats db)  db :constant]]
                        [(keyword (str "L" l "w" pl "aq")) [:byte  (alength ^bytes aq)   aq :constant]]
                        [(keyword (str "L" l "w" pl "bq")) [:byte  (alength ^bytes bq)   bq :constant]]]))
                   proj->role)))

(defn- norm-specs [m l]
  (into {} (map (fn [[nn role]]
                  (let [^floats d (:data (get (:weights m) (dec/role-name (:desc m) role l)))]
                    [(keyword (str "L" l nn)) [:float (alength d) d :constant]]))
                (norm-roles m))))

(defn- kv-specs [m l maxpos]
  (let [kvrow (* (long (:n-kv m)) (long (:head-dim m)))]
    {(keyword (str "kc" l)) [:float (* (long maxpos) kvrow) nil :state]
     (keyword (str "vc" l)) [:float (* (long maxpos) kvrow) nil :state]}))

(defn- head-specs [m qw]
  (let [{:keys [wp da db aq bq]} (get qw (dec/role-name (:desc m) :embed nil))
        ^floats fln (:data (get (:weights m) (dec/role-name (:desc m) :final-norm nil)))]
    {:lmp  [:int   (alength ^ints wp)   wp :constant]
     :lmda [:float (alength ^floats da) da :constant]
     :lmdb [:float (alength ^floats db) db :constant]
     :lmaq [:byte  (alength ^bytes aq)  aq :constant]
     :lmbq [:byte  (alength ^bytes bq)  bq :constant]
     :finalln [:float (alength fln) fln :constant]
     :logits  [:float (long (:vocab m)) nil :output]}))

(defn- io-specs [m maxpos]
  (merge {:r0 [:float (long (:d-model m)) nil :input]
          :posbuf [:long 1 nil :input] :clenbuf [:long 1 nil :input]}
         (into {} (for [l (range 1 (inc (:n-layers m)))]
                    [(keyword (str "r" l)) [:float (long (:d-model m)) nil :scratch]]))))

(defn- buffer-specs [m qw maxpos]
  (apply merge (scratch-specs m)
         ;; attention score/prob scratch: n-q rows strided by maxpos (parallel decode attention)
         {:sc [:float (* (long (:n-q m)) (long maxpos)) nil :scratch]}
         (io-specs m maxpos) (head-specs m qw)
         (concat (for [l (range (:n-layers m))] (weight-specs m qw l))
                 (for [l (range (:n-layers m))] (norm-specs m l))
                 (for [l (range (:n-layers m))] (kv-specs m l maxpos)))))

(defn- layer-theta
  "Per-layer rope base from descriptor flags: :dual = global theta on global layers
  (explicit set or every-p pattern), local otherwise; :single = one theta."
  [m l]
  (let [flags (get-in m [:desc :flags])]
    (if (= :dual (:rope flags))
      (if (cond (:global-layers flags) (contains? (:global-layers flags) l)
                (:global-layer-pattern flags) (zero? (mod (inc (long l))
                                                          (long (:global-layer-pattern flags))))
                :else true)
        (:rope-global m) (:rope-local m))
      (:rope-global m))))

(defn- scalar-args [prog kvs]
  (mapv (fn [p] (get kvs (name p))) (:all-params prog)))

(defn embed-row ^floats [m token]
  (let [d (long (:d-model m))
        ^floats emb (:data (get (:weights m) (dec/role-name (:desc m) :embed nil)))
        scale (embed-scale m) x (float-array d) base (* (long token) d)]
    (dotimes [j d] (aset x j (float (* scale (aget emb (+ base j)))))) x))

;; ---------------------------------------------------------------------------
;; S3: GPU PREFILL/EMBED mode — a whole T-token block per replay (weights read
;; once per text: the compute-bound shape where the iGPU wins). Signed q8_0
;; weights x signed i8 activations (embeddings need >=8-bit — measured; the
;; signed x signed dp4a needs no zero-point fold). T is BAKED into the program;
;; shorter texts pad (causal mask makes pad rows harmless) and pool row len-1.
;; No KV cache, no lm-head — the final residual downloads and the host does
;; final-norm + last-token pooling + L2 (a d-float job).
;; ---------------------------------------------------------------------------

(defn quantize-q8s
  "Signed q8_0 quantize+pack every linear-role weight (raster.quant.pack/quantize-one-q8:
  row-major, per-32 block d = max|w|/127, q in [-127,127], 4 bytes/int32).
  {hf-name -> {:wp :ws :in :out}}. Requires in % 32 == 0 (all supported dims are).
  EAGER and memory-bounded: strictly one tensor at a time (no chunked-lazy pmap
  realizing dozens of packed weights at once)."
  [m]
  (let [desc (:desc m)
        names+shapes (vec (for [l (range (:n-layers m)) role (:linear-roles desc)]
                            [(dec/role-name desc role l) (#'dec/linear-shape m role)]))]
    (persistent!
     (reduce (fn [acc [nm [in out]]]
               (assoc! acc nm (qpack/quantize-one-q8 (:data (get (:weights m) nm)) in out nm)))
             (transient {})
             names+shapes))))

(defn- embed-layer-form
  "Generated PREFILL layer deftm: flags decide qk-norm/sandwich/act (as gen-layer!),
  dims + T baked. Weights are 2-array signed-q8 (w{pl}p/w{pl}s)."
  [m nm T]
  (let [{:keys [d dff qd kvrow nq nkv hd group]} (dims-of m)
        T (long T)
        flags (get-in m [:desc :flags])
        qk? (boolean (:qk-norm flags))
        sand? (boolean (:sandwich-norms flags))
        bidir? (boolean (:bidirectional? flags))
        act-mul (if (= :geglu (:ffn flags)) 'nn/gelu-mul! 'nn/silu-mul!)
        go (double (get-in flags [:norm :gain-offset] 0.0))
        wp2 (fn [pl] [(symbol (str "w" pl "p")) :- (af :i) (symbol (str "w" pl "s")) :- (af :f)])
        params (vec (concat
                     ['r-in :- (af :f) 'inln :- (af :f)]
                     (when qk? ['qln :- (af :f) 'kln :- (af :f)])
                     (when sand? ['paln :- (af :f)])
                     ['pfln :- (af :f)]
                     (when sand? ['pffln :- (af :f)])
                     (mapcat wp2 ["q" "k" "v" "o" "g" "u" "d"])
                     ['h :- (af :f) 'hp :- (af :i) 'hs :- (af :f)
                      'q :- (af :f) 'k :- (af :f) 'v :- (af :f)]
                     (when qk? ['qn :- (af :f) 'kn :- (af :f)])
                     ['qr :- (af :f) 'kr :- (af :f) 'sc :- (af :f) 'at :- (af :f)
                      'ap :- (af :i) 'as :- (af :f) 'o :- (af :f)]
                     (when sand? ['o2 :- (af :f)])
                     ['xmid :- (af :f) 'f :- (af :f) 'fp :- (af :i) 'fs :- (af :f)
                      'gate :- (af :f) 'up :- (af :f) 'hh :- (af :f)
                      'hhp :- (af :i) 'hhs :- (af :f) 'down :- (af :f)]
                     (when sand? ['down2 :- (af :f)])
                     ['r-out :- (af :f)
                      'eps :- 'Double 'theta :- 'Double 'scale :- 'Double]))
        rmsT (fn [x w out rows feat] (list 'nn/rms-norm! x w out rows feat 'eps go))
        body (remove nil?
              [(rmsT 'r-in 'inln 'h T d)
               (list 'qk/quant-act-i8-rows-gpu! 'h 'hp 'hs d T)
               (list 'qk/qmatmul-i8-gemm! 'hp 'hs 'wqp 'wqs 'q d qd T)
               (list 'qk/qmatmul-i8-gemm! 'hp 'hs 'wkp 'wks 'k d kvrow T)
               (list 'qk/qmatmul-i8-gemm! 'hp 'hs 'wvp 'wvs 'v d kvrow T)
               (when qk? (rmsT 'q 'qln 'qn (* T nq) hd))
               (when qk? (rmsT 'k 'kln 'kn (* T nkv) hd))
               (list 'attn/rope-prefill! (if qk? 'qn 'q) 'qr T nq hd 'theta)
               (list 'attn/rope-prefill! (if qk? 'kn 'k) 'kr T nkv hd 'theta)
               (list (if bidir? 'attn/attn-prefill-scores-bidir! 'attn/attn-prefill-scores!)
                     'qr 'kr 'sc T nq group nkv hd 'scale)
               (list 'attn/attn-prefill-softmax! 'sc T nq)
               (list 'attn/attn-prefill-out! 'sc 'v 'at T nq group nkv hd)
               (list 'qk/quant-act-i8-rows-gpu! 'at 'ap 'as qd T)
               (list 'qk/qmatmul-i8-gemm! 'ap 'as 'wop 'wos 'o qd d T)
               (when sand? (rmsT 'o 'paln 'o2 T d))
               (list 'nn/residual-add! 'r-in (if sand? 'o2 'o) 'xmid (* T d))
               (rmsT 'xmid 'pfln 'f T d)
               (list 'qk/quant-act-i8-rows-gpu! 'f 'fp 'fs d T)
               (list 'qk/qmatmul-i8-gemm! 'fp 'fs 'wgp 'wgs 'gate d dff T)
               (list 'qk/qmatmul-i8-gemm! 'fp 'fs 'wup 'wus 'up d dff T)
               (list act-mul 'gate 'up 'hh (* T dff))
               (list 'qk/quant-act-i8-rows-gpu! 'hh 'hhp 'hhs dff T)
               (list 'qk/qmatmul-i8-gemm! 'hhp 'hhs 'wdp 'wds 'down dff d T)
               (when sand? (rmsT 'down 'pffln 'down2 T d))
               (list 'nn/residual-add! 'xmid (if sand? 'down2 'down) 'r-out (* T d))])]
    (list 'raster.core/deftm nm params :- 'Void (cons 'do body))))

(defn gen-embed-layer!
  "Get-or-create the generated PREFILL layer deftm for model m at block size T."
  [m T]
  (let [{:keys [d dff]} (dims-of m)
        nm (symbol (str "gembed-" (name (get-in m [:desc :arch])) "-" d "x" dff "-t" T "!"))]
    (eval-gen! nm (embed-layer-form m nm T))))

(defn- embed-scratch-specs [m T]
  (let [{:keys [d dff qd kvrow]} (dims-of m)
        T (long T)]
    {:h [:float (* T d) nil :scratch] :hp [:int (* T (quot d 4)) nil :scratch]
     :hs [:float (* T (quot d 32)) nil :scratch]
     :q [:float (* T qd) nil :scratch] :k [:float (* T kvrow) nil :scratch]
     :v [:float (* T kvrow) nil :scratch]
     :qn [:float (* T qd) nil :scratch] :kn [:float (* T kvrow) nil :scratch]
     :qr [:float (* T qd) nil :scratch] :kr [:float (* T kvrow) nil :scratch]
     :sc [:float (* T (long (:n-q m)) T) nil :scratch]
     :at [:float (* T qd) nil :scratch]
     :ap [:int (* T (quot qd 4)) nil :scratch] :as [:float (* T (quot qd 32)) nil :scratch]
     :o [:float (* T d) nil :scratch] :o2 [:float (* T d) nil :scratch]
     :xmid [:float (* T d) nil :scratch]
     :f [:float (* T d) nil :scratch] :fp [:int (* T (quot d 4)) nil :scratch]
     :fs [:float (* T (quot d 32)) nil :scratch]
     :gate [:float (* T dff) nil :scratch] :up [:float (* T dff) nil :scratch]
     :hh [:float (* T dff) nil :scratch]
     :hhp [:int (* T (quot dff 4)) nil :scratch] :hhs [:float (* T (quot dff 32)) nil :scratch]
     :down [:float (* T d) nil :scratch] :down2 [:float (* T d) nil :scratch]}))

(defn- embed-weight-specs [m qw l]
  (into {} (mapcat (fn [[pl role]]
                     (let [{:keys [wp ws]} (get qw (dec/role-name (:desc m) role l))]
                       [[(keyword (str "L" l "w" pl "p")) [:int (alength ^ints wp) wp :constant]]
                        [(keyword (str "L" l "w" pl "s")) [:float (alength ^floats ws) ws :constant]]]))
                   proj->role)))

(defn- compile-resident-or-throw
  "compile-gpu-program returns nil when the deftm is not a straight-line resident
  program (control flow / a reduce it can't lower to the resident graph). Binding a
  nil program silently records an EMPTY command graph -> the layer never runs, the
  residual stays zero, the whole stack outputs 0.0, and any argmax tail then breaks a
  tie by last-writer -> garbage. Fail loud here instead of miscompiling to zeros."
  ([v what] (compile-resident-or-throw v what nil))
  ([v what hint]
   ;; :on-non-resident :nil so THIS wrapper's message (with the domain-specific hint) fires,
   ;; rather than compile-gpu-program's generic binding-level throw.
   (or (pipeline/compile-gpu-program v :ze:0 :dtype :float :on-non-resident :nil)
       (throw (ex-info (str "compile-gpu-program returned nil for " what
                            " — not a resident-compilable program" (when hint (str " " hint)))
                       {:program what})))))

(defn bind-embed!
  "Compile + bind the PREFILL embed program: n-layers x generated embed layer over
  T-sized resident buffers. Returns {:sess :model :T}. Replay per text via embed-gpu."
  [m & {:keys [T qw] :or {T 128}}]
  (let [eps (:eps m) scale (:attn-scale m)
        ;; The symmetric-window mask kernel now exists on the substrate:
        ;; attn/attn-prefill-scores-windowed! (validated GPU-resident, left =
        ;; right = w gives |i-j| < w). Closing this assert means gen-embed-layer!
        ;; must pick windowed vs bidir PER LAYER (EmbeddingGemma alternates
        ;; sliding/global layers) and thread the two window scalars through the
        ;; generated program + revalidate the GPU embedder anchor at T > 512 —
        ;; left for a dedicated pass.
        _ (when (get-in m [:desc :flags :bidirectional?])
            (when-let [w (get-in m [:desc :flags :sliding-window :size])]
              (assert (<= (long T) (long w))
                      (str "bidirectional prefill with T=" T " > sliding window " w
                           " needs the symmetric-window mask (not yet emitted)"))))
        layer-var (gen-embed-layer! m T)
        layer-prog (compile-resident-or-throw layer-var (str "embed layer (T=" T ")"))
        qw (or qw (quantize-q8s m))
        d (long (:d-model m))
        norm-names (set (keys (norm-roles m)))
        specs (apply merge
                     (embed-scratch-specs m T)
                     {:r0 [:float (* (long T) d) nil :input]}
                     (into {} (for [l (range 1 (inc (:n-layers m)))]
                                [(keyword (str "r" l))
                                 [:float (* (long T) d) nil
                                  (if (= l (:n-layers m)) :output :scratch)]]))
                     (concat (for [l (range (:n-layers m))] (embed-weight-specs m qw l))
                             (for [l (range (:n-layers m))] (norm-specs m l))))
        alloc-specs (into {} (map (fn [[k v]] [k (vec (take 3 v))]) specs))
        roles (into {} (map (fn [[k v]] [k (nth v 3 :scratch)]) specs))
        layer-args (scalar-args layer-prog {"eps" eps "theta" (:rope-global m) "scale" scale})
        prog-alloc-specs (into {}
                           (for [l (range (:n-layers m))
                                 {:keys [sym size-fn]} (:allocs layer-prog)]
                             [(keyword (str "L" l "_" (name sym)))
                              [:float (long (size-fn layer-args)) nil]]))
        sess (gpu/make-session :ze:0)
        phases (atom [])]
    (gpu/alloc! sess (merge alloc-specs prog-alloc-specs))
    (swap! sess assoc :chain-roles roles)
    (doseq [l (range (:n-layers m))]
      (let [args (scalar-args layer-prog {"eps" eps "theta" (layer-theta m l) "scale" scale})]
        (doseq [step (:steps layer-prog)]
          (let [ph (keyword (str "L" l "_" (name (:phase step))))]
            (gpu/bind-step! sess (assoc step :phase ph) args (fn [a] (pbk norm-names l (name a))))
            (swap! phases conj ph)))))
    (gpu/record-graph! sess @phases :embed)
    {:sess sess :model m :T T}))

(declare embed-pool-last embed-pool-mean)

(defn embed-gpu
  "Embed token ids on the GPU prefill program: pad/truncate to T, upload the token
  embeddings, replay, download the final residual, host-side final-norm + last-token
  pool + L2 normalize. Returns float[d-model]."
  ^floats [estate ids]
  (let [{:keys [sess model T]} estate
        d (long (:d-model model))
        T (long T)
        ids (vec (take T ids))
        n (count ids)
        r0 (float-array (* T d))]
    (dotimes [i n]
      (System/arraycopy ^floats (embed-row model (nth ids i)) 0 r0 (* i d) d))
    (gpu/upload! sess :r0 r0)
    (gpu/replay! sess :embed)
    (if (= :mean (get-in model [:desc :flags :pooling]))
      (embed-pool-mean model (gpu/download sess (keyword (str "r" (:n-layers model)))) n)
      (embed-pool-last model (gpu/download sess (keyword (str "r" (:n-layers model)))) n))))

(defn- embed-pool-last
  "Last-token pooling (Qwen3-Embedding): final-norm the last real row, L2 normalize."
  ^floats [model ^floats rf n]
  (let [d (long (:d-model model))
        base (* (dec (long n)) d)
        ^floats fln (:data (get (:weights model) (dec/role-name (:desc model) :final-norm nil)))
        go (double (get-in model [:desc :flags :norm :gain-offset] 0.0))
        eps (double (:eps model))
        row (java.util.Arrays/copyOfRange rf (int base) (int (+ base d)))
        ^floats e (nn/rms-norm row fln 1 d eps go)]
    (nn/l2-normalize! e d)))

(defn load-dense-head
  "Load the sentence-transformers Dense projection weights (2_Dense/3_Dense
  model.safetensors, tensor `linear.weight`, out×in, no bias) from the model dir.
  Returns [{:w floats :out N :in M} ...] or nil if absent."
  [dir]
  (let [one (fn [sub]
              (let [f (str dir "/" sub "/model.safetensors")]
                (when (.exists (java.io.File. f))
                  (let [t (get (pretrained.safetensors/load-safetensors f) "linear.weight")
                        [out in] (:shape t)]
                    {:w (:data t) :out (long out) :in (long in)}))))]
    (vec (keep one ["2_Dense" "3_Dense"]))))

(defn- dense-apply ^floats [{:keys [^floats w out in]} ^floats x]
  (let [y (float-array out)]
    (dotimes [o out]
      (aset y o (float (loop [i 0 s 0.0]
                         (if (< i (long in))
                           (recur (inc i) (+ s (* (aget w (+ (* o (long in)) i)) (aget x i))))
                           s)))))
    y))

(defn- embed-pool-mean
  "Mean pooling over the n real rows (EmbeddingGemma): final-norm each row, mean,
  optional Dense projections (768->3072->768, no bias), L2 normalize."
  ^floats [model ^floats rf n]
  (let [d (long (:d-model model)) n (long n)
        ^floats fln (:data (get (:weights model) (dec/role-name (:desc model) :final-norm nil)))
        go (double (get-in model [:desc :flags :norm :gain-offset] 0.0))
        eps (double (:eps model))
        ^floats normed (nn/rms-norm (java.util.Arrays/copyOfRange rf 0 (int (* n d)))
                                    fln n d eps go)
        ^floats pooled (nn/mean-pool normed n d)
        ^floats e (reduce (fn [^floats x head] (dense-apply head x)) pooled (:dense model))]
    (nn/l2-normalize! e (long (alength e)))))

(defn bind-decode!
  "Compile the layer + head programs, Q4K-quantize the weights, allocate resident buffers, bind
  18 layers + head into ONE command graph. Returns a decode-state {:sess :model …}."
  [m & {:keys [maxpos layer-var head-var qw rms-style prefill-T steer]
        :or {maxpos 64 rms-style :map-void}}]
  (let [eps (:eps m) scale (:attn-scale m)
        ;; Phase-2b: `steer` = {layer-idx ^floats vec} — a per-layer activation-addition at
        ;; :resid-post (steering / concept injection). When present, generate the steer-variant
        ;; layer (an injected residual-add!) and allocate a per-layer L{l}steer :constant buffer,
        ;; zero except at target layers. Absent → identical program (no injected kernel).
        steer? (boolean (seq steer))
        d (long (:d-model m))
        layer-var (or layer-var (gen-layer! m :rms-style rms-style :steer? steer?))
        head-var  (or head-var (gen-head! m))
        norm-names (set (keys (norm-roles m)))
        fn-hint (when (= rms-style :fn) "(rms-style :fn is incomplete; use :map-void)")
        layer-prog (compile-resident-or-throw layer-var (str "layer (rms-style " rms-style ")") fn-hint)
        head-prog  (compile-resident-or-throw head-var "head")
        tail-prog  (compile-resident-or-throw #'decode-tail! "decode-tail!")
        tail-args  (scalar-args tail-prog {"vocab" (long (:vocab m)) "d" (long (:d-model m))})
        qw (or qw (gpu-quantize m))
        steer-specs (when steer?
                      (into {} (for [l (range (:n-layers m))]
                                 [(keyword (str "L" l "steer"))
                                  [:float d (or (get steer l) (float-array d)) :constant]])))
        specs (merge (buffer-specs m qw maxpos)
                     steer-specs
                     {:emb    [:float (* (long (:vocab m)) (long (:d-model m))) (prescaled-embed m) :constant]
                      :tokbuf [:int 1 nil :output]})
        alloc-specs (into {} (map (fn [[k v]] [k (vec (take 3 v))]) specs))
        roles (into {} (map (fn [[k v]] [k (nth v 3 :scratch)]) specs))
        ;; A program's :allocs are compiler-introduced scratch — Stage A's reduce-result ss__rbuf
        ;; 1-elem buffers. These must be PER-LAYER (key matches pbk's L{l}_… for __rbuf), NOT shared,
        ;; to avoid the cross-layer write-after-read hazard. The head has no reduce allocs.
        ;; Size-fns are constant, so any layer's scalar args suffice.
        layer-alloc-args (scalar-args layer-prog {"eps" eps "theta" (:rope-local m) "scale" scale
                                                  "maxpos" (long maxpos)})
        prog-alloc-specs (into {}
                           (concat
                            (for [l (range (:n-layers m))
                                  {:keys [sym size-fn]} (:allocs layer-prog)]
                              [(keyword (str "L" l "_" (name sym))) [:float (long (size-fn layer-alloc-args)) nil]])
                            (for [{:keys [sym size-fn]} (:allocs head-prog)]
                              [(keyword (name sym)) [:float (long (size-fn (scalar-args head-prog {"eps" eps}))) nil]])
                            (for [{:keys [sym size-fn]} (:allocs tail-prog)]
                              [(keyword (name sym)) [:float (long (size-fn tail-args)) nil]])))
        sess (gpu/make-session :ze:0)
        phases (atom [])]
    (gpu/alloc! sess (merge alloc-specs prog-alloc-specs))
    (swap! sess assoc :chain-roles roles)
    (doseq [l (range (:n-layers m))]
      (let [args (scalar-args layer-prog {"eps" eps "theta" (layer-theta m l) "scale" scale
                                          "maxpos" (long maxpos)})]
        (doseq [step (:steps layer-prog)]
          (let [ph (keyword (str "L" l "_" (name (:phase step))))]
            (gpu/bind-step! sess (assoc step :phase ph) args (fn [a] (pbk norm-names l (name a))))
            (swap! phases conj ph)))))
    (let [args (scalar-args head-prog {"eps" eps})]
      (doseq [step (:steps head-prog)]
        (let [ph (keyword (str "H_" (name (:phase step))))]
          (gpu/bind-step! sess (assoc step :phase ph) args (fn [a] (head-pbk (:n-layers m) (name a))))
          (swap! phases conj ph))))
    ;; decode tail (argmax + embed-gather); params map to session buffers by their own names
    (doseq [step (:steps tail-prog)]
      (let [ph (keyword (str "T_" (name (:phase step))))]
        (gpu/bind-step! sess (assoc step :phase ph) tail-args (fn [a] (keyword (name a))))
        (swap! phases conj ph)))
    (gpu/record-graph! sess @phases :decode)
    ;; ---- optional batched PREFILL graph (phase 2a) ----
    ;; The embed-layer program at block size prefill-T, bound into the SAME
    ;; session with its roped-keys/values args pointing DIRECTLY at the decode
    ;; KV caches: kr/v are [T,kvrow] row-major = exactly the first T rows of
    ;; kc{l}/vc{l} — the prefill GEMM+rope write the cache in place, zero-copy.
    ;; Causal attention within the block (the qwen3 embed program is causal),
    ;; positions 0..T-1 = absolute prompt positions. Pad rows beyond the real
    ;; prompt are harmless: causal masking keeps real rows clean, and the
    ;; rollout overwrites pad KV rows position by position as it generates.
    ;; Q8 weights (the prefill kernels are i8-gemm), norms shared with decode.
    (when prefill-T
      (let [Tp (long prefill-T)
            _ (assert (<= Tp (long maxpos)) "prefill-T must fit in maxpos")
            {:keys [d dff qd kvrow]} (dims-of m)
            pvar (gen-embed-layer! m Tp)
            pprog (pipeline/compile-gpu-program pvar :ze:0 :dtype :float)
            q8 (quantize-q8s m)
            pscratch (into {} (map (fn [[k v]] [(keyword (str "p_" (name k))) v])
                                   (dissoc (embed-scratch-specs m Tp) :k :v)))
            ;; k (pre-rope) stays scratch; kr + v go to the caches
            presid (into {} (for [l (range (inc (:n-layers m)))]
                              [(keyword (str "pr" l)) [:float (* Tp (long d)) nil]]))
            pweights (into {}
                       (mapcat (fn [l]
                                 (mapcat (fn [[pl role]]
                                           (let [{:keys [wp ws]} (get q8 (dec/role-name (:desc m) role l))]
                                             [[(keyword (str "PL" l "w" pl "p")) [:int (alength ^ints wp) wp]]
                                              [(keyword (str "PL" l "w" pl "s")) [:float (alength ^floats ws) ws]]]))
                                         proj->role))
                               (range (:n-layers m))))
            pargs (scalar-args pprog {"eps" (:eps m) "theta" (:rope-local m)
                                      "scale" (:attn-scale m)})
            pallocs (into {}
                      (for [l (range (:n-layers m))
                            {:keys [sym size-fn]} (:allocs pprog)]
                        [(keyword (str "PL" l "_" (name sym))) [:float (long (size-fn pargs)) nil]]))
            ppbk (fn [l]
                   (fn [a]
                     (let [pn (name a)]
                       (cond (= pn "r-in")  (keyword (str "pr" l))
                             (= pn "r-out") (keyword (str "pr" (inc l)))
                             (= pn "kr")    (keyword (str "kc" l))   ; ← the cache
                             (= pn "v")     (keyword (str "vc" l))   ; ← the cache
                             (= pn "k")     :p_k                     ; pre-rope scratch
                             (norm-names pn) (keyword (str "L" l pn)) ; SHARED with decode
                             (= \w (first pn)) (keyword (str "PL" l pn))
                             (re-find #"__rbuf" pn) (keyword (str "PL" l "_" pn))
                             :else (keyword (str "p_" pn))))))
            pphases (atom [])]
        (gpu/alloc! sess (merge pscratch presid pweights pallocs
                                {:p_k [:float (* Tp (long kvrow)) nil]}))
        (doseq [l (range (:n-layers m))]
          (let [args (scalar-args pprog {"eps" (:eps m) "theta" (layer-theta m l)
                                         "scale" (:attn-scale m)})]
            (doseq [step (:steps pprog)]
              (let [ph (keyword (str "PL" l "_" (name (:phase step))))]
                (gpu/bind-step! sess (assoc step :phase ph) args (ppbk l))
                (swap! pphases conj ph)))))
        (gpu/record-graph! sess @pphases :prefill)))
    {:sess sess :model m :maxpos maxpos :prefill-T prefill-T}))

(defn prefill-rows!
  "Batched prompt prefill: upload the [T,d] input rows (token embeddings with
  any multimodal rows already spliced; zero-pad past the real prompt) and
  replay the :prefill graph — one pass over all layers fills the decode KV
  caches for positions 0..T-1. Requires bind-decode! with :prefill-T."
  [dstate ^floats rows]
  (let [{:keys [sess prefill-T model]} dstate]
    (when-not prefill-T
      (throw (ex-info "dstate was bound without :prefill-T" {})))
    (assert (= (alength rows) (* (long prefill-T) (long (:d-model model)))))
    (gpu/upload! sess :pr0 rows)
    (gpu/replay! sess :prefill)))

(defn decode-row!
  "One resident-decode step from an ARBITRARY input row (float[d-model]): write absolute
  `pos` + the row, replay the graph, return the logits (float[vocab]). This is the
  multimodal seam — audio/vision projector rows splice into the KV stream exactly like
  token embeddings (Qwen3-ASR feeds AuT projector rows at audio positions)."
  ^floats [dstate ^floats row pos]
  (let [{:keys [sess]} dstate]
    (gpu/upload! sess :posbuf (long-array [pos]))
    (gpu/upload! sess :clenbuf (long-array [(inc pos)]))
    (gpu/upload! sess :r0 row)
    (gpu/replay! sess :decode)
    (gpu/download sess :logits)))

(defn decode-token!
  "One resident-decode step: write absolute `pos` + the embedding of `token`, replay the graph,
  return the logits (float[vocab]). The per-token cost = upload posbuf/clenbuf/r0 + graph replay +
  logits download."
  ^floats [dstate token pos]
  (decode-row! dstate (embed-row (:model dstate) token) pos))

(defn forward-logits
  "Replay the resident decode over `tokens` (absolute positions 0..); return final-pos logits."
  ^floats [dstate tokens]
  (loop [p 0 last nil]
    (if (< p (count tokens))
      (recur (inc p) (decode-token! dstate (nth tokens p) p))
      last)))

(defn gpu-hidden-states
  "GPU analogue of decoder/hidden-states (Phase-2a read tap): per-layer resid-post +
  final hidden for the LAST token of `ids`, read straight from the resident residual
  buffers. The residual stream r1..r_n is already allocated as separate per-layer
  resident buffers (pbk maps r-out -> r{l+1}), so after the last token's replay they
  hold that token's per-layer resid-post — reading them is a plain download, NO
  de-fusion and NO extra kernels (the tap is free at :resid-post on GPU). Returns
  {:layers [float[d] × n-layers] :final float[d]} to match the CPU seam."
  [dstate ids]
  (let [{:keys [sess model]} dstate
        n (long (:n-layers model)) d (long (:d-model model))
        ids (vec ids) P (count ids)]
    (doseq [p (range P)] (decode-token! dstate (nth ids p) p)) ; prime through the last token
    {:layers (mapv (fn [l] (gpu/download sess (keyword (str "r" (inc l))))) (range n))
     ;; :fh is the final-normed hidden (padded to the quant block); slice to d-model.
     :final (java.util.Arrays/copyOf ^floats (gpu/download sess :fh) (int d))}))

(defn generate-gpu
  "Greedy-generate `n` tokens after `prompt` (vector of token ids). Returns generated ids.
  Per-token host round-trip: logits download + CPU argmax + embedding upload (the tail's
  on-device r0 write is harmlessly overwritten). Kept as the reference path — prefer
  generate-resident."
  [dstate prompt n]
  (let [p0 (dec (count prompt))]
    (doseq [p (range p0)] (decode-token! dstate (nth prompt p) p)) ;; prime prompt
    (loop [p p0 tok (last prompt) out []]
      (if (< (count out) n)
        (let [nt (arr/argmax (decode-token! dstate tok p))]
          (recur (inc p) nt (conj out nt)))
        out))))

(defn generate-resident
  "Greedy autoregressive rollout fully on-device: after priming the prompt, each step uploads
  only the 16-byte position, replays the graph (layers + head + argmax/embed-gather tail), and
  downloads the 4-byte token id — the tail writes the next r0 on-device, so no logits download
  and no embedding upload. Rolls out up to `max-new` tokens, stopping early when the generated
  id is in `eos-ids` (the stop token is included in the result) or when the KV cache (maxpos)
  is full. Returns the vector of generated ids."
  [dstate prompt max-new & {:keys [eos-ids] :or {eos-ids #{}}}]
  (let [{:keys [sess model maxpos]} dstate
        p0 (dec (count prompt))]
    (doseq [p (range p0)] (decode-token! dstate (nth prompt p) p)) ;; prime prompt
    (gpu/upload! sess :r0 (embed-row model (last prompt)))
    (loop [p p0 out []]
      (if (and (< (count out) max-new) (< p (long maxpos)))
        (do (gpu/upload! sess :posbuf (long-array [p]))
            (gpu/upload! sess :clenbuf (long-array [(inc p)]))
            (gpu/replay! sess :decode)
            (let [t (aget ^ints (gpu/download sess :tokbuf) 0)
                  out (conj out t)]
              (if (contains? eos-ids t) out (recur (inc p) out))))
        out))))

