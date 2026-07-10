(ns pretrained.decoder-gpu-oracle-test
  "Oracle for the descriptor-GENERATED GPU layer/head (decoder-gpu/gen-layer!,
  gen-head!): the hand-written gemma-3-270m deftms this generator was validated
  against (ported from /tmp gcl2!/gcomp-head!, decode oracle 236761) are FROZEN
  here as fixtures, and the generated deftms must reproduce them bit-exactly on
  identical seeded inputs.

  Model-free and CPU-only: both the fixture and the generated deftm execute the
  same substrate kernels (raster.quant.kernels-k / raster.dl.{nn,attention}) on
  the JVM over random-but-seeded buffers of the real gemma-3-270m shapes — the
  claim under test is program equivalence (same ops, same order, same baked
  dims), which is independent of real weights. ^:anchors-tagged only because it
  is heavy (a few seconds of scalar quantized matmul incl. the 262144-vocab
  head), not because it needs weights or a GPU: run with
    clojure -A:dev -M:test -i :anchors -v pretrained.decoder-gpu-oracle-test"
  (:require [clojure.test :refer [deftest is testing]]
            [pretrained.decoder-gpu :as dgpu]
            [pretrained.arch.gemma3 :as gemma3]
            [raster.core]
            [raster.par]
            [raster.numeric]
            [raster.arrays]
            [raster.dl.nn :as nn]
            [raster.dl.attention :as attn]
            [raster.quant.kernels-k :as qk]))

;; ---------------------------------------------------------------------------
;; FROZEN fixtures — the validated hand-written gemma-3-270m layer + head,
;; verbatim from pretrained.decoder-gpu prior to their deletion (the generated
;; path replaced them in production). Do not "clean up": these are the oracle.
;; ---------------------------------------------------------------------------

(raster.core/deftm gemma-layer-oracle!
  [r-in :- (Array float) inln :- (Array float) qln :- (Array float) kln :- (Array float)
   paln :- (Array float) pfln :- (Array float) pffln :- (Array float)
   wqp :- (Array int) wqda :- (Array float) wqdb :- (Array float) wqaq :- (Array byte) wqbq :- (Array byte)
   wkp :- (Array int) wkda :- (Array float) wkdb :- (Array float) wkaq :- (Array byte) wkbq :- (Array byte)
   wvp :- (Array int) wvda :- (Array float) wvdb :- (Array float) wvaq :- (Array byte) wvbq :- (Array byte)
   wop :- (Array int) woda :- (Array float) wodb :- (Array float) woaq :- (Array byte) wobq :- (Array byte)
   wgp :- (Array int) wgda :- (Array float) wgdb :- (Array float) wgaq :- (Array byte) wgbq :- (Array byte)
   wup :- (Array int) wuda :- (Array float) wudb :- (Array float) wuaq :- (Array byte) wubq :- (Array byte)
   wdp :- (Array int) wdda :- (Array float) wddb :- (Array float) wdaq :- (Array byte) wdbq :- (Array byte)
   kc :- (Array float) vc :- (Array float)
   h :- (Array float) qinp :- (Array int) qins :- (Array float) qinb :- (Array int)
   q :- (Array float) k :- (Array float) v :- (Array float) qn :- (Array float) kn :- (Array float)
   qr :- (Array float) kr :- (Array float) sc :- (Array float) at :- (Array float)
   qap :- (Array int) qas :- (Array float) qab :- (Array int)
   o :- (Array float) o2 :- (Array float) xmid :- (Array float)
   f :- (Array float) qfp :- (Array int) qfs :- (Array float) qfb :- (Array int)
   gate :- (Array float) up :- (Array float) hh :- (Array float)
   qhp :- (Array int) qhs :- (Array float) qhb :- (Array int)
   down :- (Array float) down2 :- (Array float) r-out :- (Array float)
   posbuf :- (Array long) clenbuf :- (Array long) submax :- (Array float)
   maxpos :- Long eps :- Double theta :- Double scale :- Double] :- Void
  (do
    (nn/rms-norm! r-in inln h 1 640 eps 1.0)
    (qk/quant-act-q8k-gpu! h qinp qins qinb submax 3)
    (qk/qmatmul-q4k-dp4a! qinp qins qinb wqp wqda wqdb wqaq wqbq q 768 1024)
    (qk/qmatmul-q4k-dp4a! qinp qins qinb wkp wkda wkdb wkaq wkbq k 768 256)
    (qk/qmatmul-q4k-dp4a! qinp qins qinb wvp wvda wvdb wvaq wvbq v 768 256)
    (nn/rms-norm! q qln qn 4 256 eps 1.0)
    (nn/rms-norm! k kln kn 1 256 eps 1.0)
    (attn/rope-pos-buf! qn qr 4 256 theta posbuf)
    (attn/rope-pos-buf! kn kr 1 256 theta posbuf)
    (attn/kv-append-buf! kr kc 256 posbuf)
    (attn/kv-append-buf! v vc 256 posbuf)
    (attn/gqa-decode-attention-buf! qr kc vc at sc clenbuf 4 4 1 256 maxpos (float scale))
    (qk/quant-act-q8k-gpu! at qap qas qab submax 4)
    (qk/qmatmul-q4k-dp4a! qap qas qab wop woda wodb woaq wobq o 1024 640)
    (nn/rms-norm! o paln o2 1 640 eps 1.0)
    (nn/residual-add! r-in o2 xmid 640)
    (nn/rms-norm! xmid pfln f 1 640 eps 1.0)
    (qk/quant-act-q8k-gpu! f qfp qfs qfb submax 3)
    (qk/qmatmul-q4k-dp4a! qfp qfs qfb wgp wgda wgdb wgaq wgbq gate 768 2048)
    (qk/qmatmul-q4k-dp4a! qfp qfs qfb wup wuda wudb wuaq wubq up 768 2048)
    (nn/gelu-mul! gate up hh 2048)
    (qk/quant-act-q8k-gpu! hh qhp qhs qhb submax 8)
    (qk/qmatmul-q4k-dp4a! qhp qhs qhb wdp wdda wddb wdaq wdbq down 2048 640)
    (nn/rms-norm! down pffln down2 1 640 eps 1.0)
    (nn/residual-add! xmid down2 r-out 640)))

;; Stage-A (:rms-style :fn) fixture: 1-row norms via the substrate's
;; nn/rms-norm-1row! (the upstreamed rms-norm-fn!); q-norm rows=4 keeps rms-norm!.
(raster.core/deftm gemma-layer-fn-oracle!
  [r-in :- (Array float) inln :- (Array float) qln :- (Array float) kln :- (Array float)
   paln :- (Array float) pfln :- (Array float) pffln :- (Array float)
   wqp :- (Array int) wqda :- (Array float) wqdb :- (Array float) wqaq :- (Array byte) wqbq :- (Array byte)
   wkp :- (Array int) wkda :- (Array float) wkdb :- (Array float) wkaq :- (Array byte) wkbq :- (Array byte)
   wvp :- (Array int) wvda :- (Array float) wvdb :- (Array float) wvaq :- (Array byte) wvbq :- (Array byte)
   wop :- (Array int) woda :- (Array float) wodb :- (Array float) woaq :- (Array byte) wobq :- (Array byte)
   wgp :- (Array int) wgda :- (Array float) wgdb :- (Array float) wgaq :- (Array byte) wgbq :- (Array byte)
   wup :- (Array int) wuda :- (Array float) wudb :- (Array float) wuaq :- (Array byte) wubq :- (Array byte)
   wdp :- (Array int) wdda :- (Array float) wddb :- (Array float) wdaq :- (Array byte) wdbq :- (Array byte)
   kc :- (Array float) vc :- (Array float)
   h :- (Array float) qinp :- (Array int) qins :- (Array float) qinb :- (Array int)
   q :- (Array float) k :- (Array float) v :- (Array float) qn :- (Array float) kn :- (Array float)
   qr :- (Array float) kr :- (Array float) sc :- (Array float) at :- (Array float)
   qap :- (Array int) qas :- (Array float) qab :- (Array int)
   o :- (Array float) o2 :- (Array float) xmid :- (Array float)
   f :- (Array float) qfp :- (Array int) qfs :- (Array float) qfb :- (Array int)
   gate :- (Array float) up :- (Array float) hh :- (Array float)
   qhp :- (Array int) qhs :- (Array float) qhb :- (Array int)
   down :- (Array float) down2 :- (Array float) r-out :- (Array float)
   posbuf :- (Array long) clenbuf :- (Array long) submax :- (Array float)
   maxpos :- Long eps :- Double theta :- Double scale :- Double] :- Void
  (do
    (nn/rms-norm-1row! r-in inln h 640 eps 1.0)
    (qk/quant-act-q8k-gpu! h qinp qins qinb submax 3)
    (qk/qmatmul-q4k-dp4a! qinp qins qinb wqp wqda wqdb wqaq wqbq q 768 1024)
    (qk/qmatmul-q4k-dp4a! qinp qins qinb wkp wkda wkdb wkaq wkbq k 768 256)
    (qk/qmatmul-q4k-dp4a! qinp qins qinb wvp wvda wvdb wvaq wvbq v 768 256)
    (nn/rms-norm! q qln qn 4 256 eps 1.0)
    (nn/rms-norm-1row! k kln kn 256 eps 1.0)
    (attn/rope-pos-buf! qn qr 4 256 theta posbuf)
    (attn/rope-pos-buf! kn kr 1 256 theta posbuf)
    (attn/kv-append-buf! kr kc 256 posbuf)
    (attn/kv-append-buf! v vc 256 posbuf)
    (attn/gqa-decode-attention-buf! qr kc vc at sc clenbuf 4 4 1 256 maxpos (float scale))
    (qk/quant-act-q8k-gpu! at qap qas qab submax 4)
    (qk/qmatmul-q4k-dp4a! qap qas qab wop woda wodb woaq wobq o 1024 640)
    (nn/rms-norm-1row! o paln o2 640 eps 1.0)
    (nn/residual-add! r-in o2 xmid 640)
    (nn/rms-norm-1row! xmid pfln f 640 eps 1.0)
    (qk/quant-act-q8k-gpu! f qfp qfs qfb submax 3)
    (qk/qmatmul-q4k-dp4a! qfp qfs qfb wgp wgda wgdb wgaq wgbq gate 768 2048)
    (qk/qmatmul-q4k-dp4a! qfp qfs qfb wup wuda wudb wuaq wubq up 768 2048)
    (nn/gelu-mul! gate up hh 2048)
    (qk/quant-act-q8k-gpu! hh qhp qhs qhb submax 8)
    (qk/qmatmul-q4k-dp4a! qhp qhs qhb wdp wdda wddb wdaq wdbq down 2048 640)
    (nn/rms-norm-1row! down pffln down2 640 eps 1.0)
    (nn/residual-add! xmid down2 r-out 640)))

(raster.core/deftm gemma-head-oracle!
  [r-fin :- (Array float) finalln :- (Array float) fh :- (Array float)
   hqp :- (Array int) hqs :- (Array float) hqb :- (Array int) submax :- (Array float)
   lmp :- (Array int) lmda :- (Array float) lmdb :- (Array float) lmaq :- (Array byte) lmbq :- (Array byte)
   logits :- (Array float) eps :- Double] :- Void
  (do
    (nn/rms-norm! r-fin finalln fh 1 640 eps 1.0)
    (qk/quant-act-q8k-gpu! fh hqp hqs hqb submax 3)
    (qk/qmatmul-q4k-dp4a! hqp hqs hqb lmp lmda lmdb lmaq lmbq logits 768 262144)))

;; ---------------------------------------------------------------------------
;; Seeded buffer construction (per-name deterministic; two builds are identical)
;; ---------------------------------------------------------------------------

(defn- rnd ^java.util.Random [nm] (java.util.Random. (long (+ 987654321 (hash nm)))))

(defn- mk [nm [kind n]]
  (case kind
    ;; activations / residual / kv-cache rows: small floats
    :act (let [r (rnd nm) a (float-array n)]
           (dotimes [i n] (aset a i (float (- (.nextDouble r) 0.5)))) a)
    ;; norm weights: gemma gain-offset 1.0 applies (1 + w) — keep w small
    :nw  (let [r (rnd nm) a (float-array n)]
           (dotimes [i n] (aset a i (float (* 0.1 (- (.nextDouble r) 0.5))))) a)
    ;; quant scales (da/db): small positive
    :qs  (let [r (rnd nm) a (float-array n)]
           (dotimes [i n] (aset a i (float (+ 0.001 (* 0.01 (.nextDouble r)))))) a)
    ;; packed weights: arbitrary int/byte patterns (both programs decode identically)
    :ri  (let [r (rnd nm) a (int-array n)]
           (dotimes [i n] (aset a i (.nextInt r))) a)
    :rb  (let [b (byte-array n)] (.nextBytes (rnd nm) b) b)
    ;; scratch: zeroed (overwritten by the programs)
    :zf  (float-array n)
    :zi  (int-array n)
    :lv  (long-array [(long n)])
    :s   n))

(def ^:private W-dims ;; proj -> [out in-padded] (gemma-3-270m, in padded to 256)
  {"q" [1024 768] "k" [256 768] "v" [256 768] "o" [640 1024]
   "g" [2048 768] "u" [2048 768] "d" [640 2048]})

(defn- wspec [pl]
  (let [[out in] (W-dims pl)]
    [[(str "w" pl "p")  [:ri (* out (quot in 8))]]
     [(str "w" pl "da") [:qs (* out (quot in 256))]]
     [(str "w" pl "db") [:qs (* out (quot in 256))]]
     [(str "w" pl "aq") [:rb (* out (quot in 32))]]
     [(str "w" pl "bq") [:rb (* out (quot in 32))]]]))

(def ^:private layer-spec
  (concat
   [["r-in" [:act 640]] ["inln" [:nw 640]] ["qln" [:nw 256]] ["kln" [:nw 256]]
    ["paln" [:nw 640]] ["pfln" [:nw 640]] ["pffln" [:nw 640]]]
   (mapcat wspec ["q" "k" "v" "o" "g" "u" "d"])
   [["kc" [:act 2048]] ["vc" [:act 2048]]           ;; maxpos=8 x kvrow=256
    ["h" [:zf 768]] ["qinp" [:zi 192]] ["qins" [:zf 3]] ["qinb" [:zi 24]]
    ["q" [:zf 1024]] ["k" [:zf 256]] ["v" [:zf 256]] ["qn" [:zf 1024]] ["kn" [:zf 256]]
    ["qr" [:zf 1024]] ["kr" [:zf 256]] ["sc" [:zf 32]] ["at" [:zf 1024]]
    ["qap" [:zi 256]] ["qas" [:zf 4]] ["qab" [:zi 32]]
    ["o" [:zf 640]] ["o2" [:zf 640]] ["xmid" [:zf 640]]
    ["f" [:zf 768]] ["qfp" [:zi 192]] ["qfs" [:zf 3]] ["qfb" [:zi 24]]
    ["gate" [:zf 2048]] ["up" [:zf 2048]] ["hh" [:zf 2048]]
    ["qhp" [:zi 512]] ["qhs" [:zf 8]] ["qhb" [:zi 64]]
    ["down" [:zf 640]] ["down2" [:zf 640]] ["r-out" [:zf 640]]
    ["posbuf" [:lv 3]] ["clenbuf" [:lv 4]] ["submax" [:zf 64]]
    ["maxpos" [:s 8]] ["eps" [:s 1.0e-6]] ["theta" [:s 10000.0]] ["scale" [:s 0.0625]]]))

;; the fixture AND the generated layer share this exact param order (verified
;; against layer-form's params vector for qk-norm+sandwich flags, no steer)
(def ^:private layer-args (mapv first layer-spec))

(def ^:private head-spec
  [["r-fin" [:act 640]] ["finalln" [:nw 640]] ["fh" [:zf 768]]
   ["hqp" [:zi 192]] ["hqs" [:zf 3]] ["hqb" [:zi 24]] ["submax" [:zf 64]]
   ["lmp" [:ri (* 262144 96)]] ["lmda" [:qs (* 262144 3)]] ["lmdb" [:qs (* 262144 3)]]
   ["lmaq" [:rb (* 262144 24)]] ["lmbq" [:rb (* 262144 24)]]
   ["logits" [:zf 262144]] ["eps" [:s 1.0e-6]]])

(def ^:private head-oracle-args (mapv first head-spec))
;; head-form orders submax after logits (before eps)
(def ^:private head-gen-args
  ["r-fin" "finalln" "fh" "hqp" "hqs" "hqb"
   "lmp" "lmda" "lmdb" "lmaq" "lmbq" "logits" "submax" "eps"])

(defn- build [spec] (into {} (map (fn [[nm k]] [nm (mk nm k)])) spec))

(def ^:private fake-270m
  ;; gemma-3-270m dims + the real gemma3 descriptor — what gen-layer!/gen-head!
  ;; consume; no weights, no tokenizer.
  {:desc gemma3/descriptor
   :d-model 640 :d-ff 2048 :n-q 4 :n-kv 1 :head-dim 256 :vocab 262144})

(defn- feq? [a b] (java.util.Arrays/equals ^floats a ^floats b))
(defn- finite-nonzero? [^floats a]
  (and (every? #(Float/isFinite (aget a (int %))) (range (alength a)))
       (some #(not (zero? (aget a (int %)))) (range (alength a)))))

(defn- run-layer [layer-fn bufs]
  (apply layer-fn (map bufs layer-args))
  bufs)

(deftest ^:anchors generated-layer-reproduces-hand-written
  (testing ":map-void generated layer == frozen hand-written gemma-layer!"
    (let [b1 (run-layer gemma-layer-oracle! (build layer-spec))
          v (dgpu/gen-layer! fake-270m :rms-style :map-void)
          b2 (run-layer (deref v) (build layer-spec))]
      (is (finite-nonzero? (b1 "r-out")) "oracle output non-degenerate")
      (is (feq? (b1 "r-out") (b2 "r-out")) "residual out bit-exact")
      (is (feq? (b1 "kc") (b2 "kc")) "K cache bit-exact")
      (is (feq? (b1 "vc") (b2 "vc")) "V cache bit-exact")))
  (testing ":fn (Stage-A rms) generated layer == frozen hand-written gemma-layer-fn!"
    (let [b1 (run-layer gemma-layer-fn-oracle! (build layer-spec))
          v (dgpu/gen-layer! fake-270m :rms-style :fn)
          b2 (run-layer (deref v) (build layer-spec))]
      (is (finite-nonzero? (b1 "r-out")) "oracle output non-degenerate")
      (is (feq? (b1 "r-out") (b2 "r-out")) "residual out bit-exact")
      (is (feq? (b1 "kc") (b2 "kc")) "K cache bit-exact"))))

(deftest ^:anchors generated-head-reproduces-hand-written
  (let [b1 (build head-spec)
        _ (apply gemma-head-oracle! (map b1 head-oracle-args))
        v (dgpu/gen-head! fake-270m)
        b2 (build head-spec)
        _ (apply (deref v) (map b2 head-gen-args))]
    (is (finite-nonzero? (b1 "logits")) "oracle logits non-degenerate")
    (is (feq? (b1 "logits") (b2 "logits")) "logits bit-exact")))
