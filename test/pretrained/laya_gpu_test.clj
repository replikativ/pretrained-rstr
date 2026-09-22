(ns pretrained.laya-gpu-test
  (:require [clojure.test :refer [deftest is]]
            [pretrained.decision.laya :as laya]
            [pretrained.decision.laya-gpu :as laya-gpu]))

(defn- random-floats [n seed scale]
  (let [out (float-array n)
        random (java.util.Random. seed)]
    (dotimes [i n]
      (aset out i (float (* scale (- (.nextDouble random) 0.5)))))
    out))

(defn- max-error [^floats a ^floats b]
  (reduce max 0.0 (map #(Math/abs (double (- %1 %2))) a b)))

(defn- tensor [data] {:data data})

(deftest ^:anchors resident-head-preserves-cpu-laya-semantics
  (let [seq-len 2 d 64 ffn (* 4 d) heads 1 head-dim 64
        prefix "head.layers.0."
        norm1-w (float-array d 1.0)
        norm1-b (random-floats d 1 0.02)
        qkv-w (random-floats (* 3 d d) 2 0.02)
        qkv-b (random-floats (* 3 d) 3 0.02)
        out-w (random-floats (* d d) 4 0.02)
        out-b (random-floats d 5 0.02)
        norm2-w (float-array d 1.0)
        norm2-b (random-floats d 6 0.02)
        linear1-w (random-floats (* ffn d) 7 0.02)
        linear1-b (random-floats ffn 8 0.02)
        linear2-w (random-floats (* d ffn) 9 0.02)
        linear2-b (random-floats d 10 0.02)
        type-emb (random-floats (* 3 d) 11 0.02)
        weights {(str prefix "norm1.weight") (tensor norm1-w)
                 (str prefix "norm1.bias") (tensor norm1-b)
                 (str prefix "self_attn.in_proj_weight") (tensor qkv-w)
                 (str prefix "self_attn.in_proj_bias") (tensor qkv-b)
                 (str prefix "self_attn.out_proj.weight") (tensor out-w)
                 (str prefix "self_attn.out_proj.bias") (tensor out-b)
                 (str prefix "norm2.weight") (tensor norm2-w)
                 (str prefix "norm2.bias") (tensor norm2-b)
                 (str prefix "linear1.weight") (tensor linear1-w)
                 (str prefix "linear1.bias") (tensor linear1-b)
                 (str prefix "linear2.weight") (tensor linear2-w)
                 (str prefix "linear2.bias") (tensor linear2-b)
                 "type_emb.weight" (tensor type-emb)}
        agent (laya/map->LayaAgent
               {:encoder {:d-model d :weights weights} :config {}})
        x (random-floats (* seq-len d) 12 0.2)
        question-type 1
        typed (#'laya/add-question-type agent x seq-len question-type)
        expected (#'laya/head-layer agent typed seq-len 0)
        type-row (laya-gpu/question-type-embedding agent question-type)
        type-values (float-array (* seq-len d))
        _ (dotimes [row seq-len]
            (System/arraycopy type-row 0 type-values (* row d) d))
        actual (apply laya-gpu/laya-resident-first-head
                      [x type-values norm1-w norm1-b qkv-w qkv-b out-w out-b
                       norm2-w norm2-b linear1-w linear1-b linear2-w linear2-b
                       seq-len d ffn heads head-dim 1.0e-5 (/ 1.0 8.0)])]
    (is (< (max-error expected actual) 1.0e-5))))
