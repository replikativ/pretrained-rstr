(ns pretrained.modernbert-test
  (:require [clojure.test :refer [deftest is testing]]
            [pretrained.arch.modernbert :as mb]))

(defn- random-floats [n seed]
  (let [out (float-array n)
        random (java.util.Random. seed)]
    (dotimes [i n]
      (aset out i (float (* 0.2 (- (.nextDouble random) 0.5)))))
    out))

(defn- max-error [^floats a ^floats b]
  (reduce max 0.0 (map #(Math/abs (double (- %1 %2))) a b)))

(deftest resident-blocks-preserve-modernbert-semantics
  (let [seq-len 4 d-model 8 d-ff 16 n-heads 2 head-dim 4
        x (random-floats (* seq-len d-model) 1)
        attn-norm (float-array (map float [0.9 1.1 0.8 1.2 0.95 1.05 0.85 1.15]))
        mlp-norm (float-array d-model 1.0)
        zero-bias (float-array d-model)
        wqkv (random-floats (* 3 d-model d-model) 2)
        wo (random-floats (* d-model d-model) 3)
        wi (random-floats (* 2 d-ff d-model) 4)
        w-mlp-out (random-floats (* d-model d-ff) 5)
        tail [seq-len d-model d-ff n-heads head-dim 10000.0 0.5 1.0e-5 seq-len]]
    (testing "first layer omits only the pre-attention normalization"
      (let [resident (apply mb/modernbert-resident-first-block
                            (concat [x mlp-norm zero-bias wqkv wo wi w-mlp-out] tail))
            reference (apply mb/modernbert-block
                             (concat [x zero-bias mlp-norm zero-bias
                                      wqkv wo wi w-mlp-out]
                                     tail [0]))]
        (is (< (max-error resident reference) 1.0e-6))))
    (testing "later layer preserves semantics with chunk-parallel LayerNorm"
      (let [resident (apply mb/modernbert-resident-block
                            (concat [x attn-norm mlp-norm zero-bias
                                     wqkv wo wi w-mlp-out]
                                    tail))
            reference (apply mb/modernbert-block
                             (concat [x attn-norm mlp-norm zero-bias
                                      wqkv wo wi w-mlp-out]
                                     tail [1]))]
        (is (< (max-error resident reference) 1.0e-6))))))
