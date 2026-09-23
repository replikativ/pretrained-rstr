(ns pretrained.modernbert-test
  (:require [clojure.test :refer [deftest is testing]]
            [pretrained.arch.modernbert :as mb]
            [raster.dl.array-ops :as ops]
            [raster.dl.nn :as nn]))

(defn- random-floats [n seed]
  (let [out (float-array n)
        random (java.util.Random. seed)]
    (dotimes [i n]
      (aset out i (float (* 0.2 (- (.nextDouble random) 0.5)))))
    out))

(defn- max-error [^floats a ^floats b]
  (reduce max 0.0 (map #(Math/abs (double (- %1 %2))) a b)))

(deftest strided-projection-fields-preserve-dense-results
  (testing "Q/K/V fields embedded in one projection match dense attention"
    (let [nrows 3 heads 2 head-dim 2 width (* heads head-dim)
          stride 15 q-offset 1 k-offset 6 v-offset 11
          q (random-floats (* nrows width) 11)
          k (random-floats (* nrows width) 12)
          v (random-floats (* nrows width) 13)
          packed (float-array (* nrows stride))
          _ (doseq [row (range nrows)
                    col (range width)]
              (aset packed (+ (* row stride) q-offset col)
                    (aget q (+ (* row width) col)))
              (aset packed (+ (* row stride) k-offset col)
                    (aget k (+ (* row width) col)))
              (aset packed (+ (* row stride) v-offset col)
                    (aget v (+ (* row width) col))))
          dense-scores (float-array (* heads nrows nrows))
          strided-scores (float-array (* heads nrows nrows))
          dense-out (float-array (* nrows width))
          strided-out (float-array (* nrows width))]
      (mb/attention! q k v dense-scores dense-out nrows heads head-dim 0.5 0)
      (mb/attention-strided! packed packed packed strided-scores strided-out
                             nrows heads head-dim 0.5 0
                             stride q-offset stride k-offset stride v-offset)
      (is (< (max-error dense-scores strided-scores) 1.0e-7))
      (is (< (max-error dense-out strided-out) 1.0e-7))))
  (testing "fused strided GeGLU matches the materialized operations"
    (let [rows 3 width 5 stride (* 2 width)
          fused (random-floats (* rows stride) 14)
          activated (ops/slice-strided-2d fused rows stride 0 width)
          gate (ops/slice-strided-2d fused rows stride width width)
          _ (nn/gelu-erf! activated activated (* rows width))
          expected (nn/hadamard activated gate (* rows width))
          actual (float-array (* rows width))]
      (mb/gelu-erf-mul-strided! fused actual rows stride 0 width width)
      (is (< (max-error expected actual) 1.0e-7)))))

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
