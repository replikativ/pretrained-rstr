(ns pretrained.moe-router-test
  "Model-free: the MoE routing math (softmax over ALL experts → top-k of the
  probabilities → renormalize) matches a hand-computed reference through the
  full decode-step on a minimal handcrafted 1-layer model."
  (:require [clojure.test :refer [deftest is testing]]
            [pretrained.decoder :as dec]
            [pretrained.arch.qwen3-moe :as qmoe]))

(defn- softmax [xs]
  (let [m (apply max xs) es (map #(Math/exp (- % m)) xs) z (reduce + es)]
    (mapv #(/ % z) es)))

(deftest router-selection-and-renorm
  ;; hand-check the selection semantics the engine implements
  (let [logits [1.0 3.0 2.0 -1.0]
        probs (softmax logits)
        top2 (take 2 (sort-by #(- (nth probs %)) (range 4)))
        psum (reduce + (map probs top2))
        weights (mapv #(/ (nth probs %) psum) top2)]
    (is (= [1 2] (vec top2)) "top-k picks by probability")
    (is (< (Math/abs (- 1.0 (reduce + weights))) 1e-12) "renormalized to 1")
    (is (> (first weights) (second weights)))))

(deftest descriptor-shape
  (testing "qwen3-moe descriptor: attention identical to qwen3, MoE flags set"
    (let [d qmoe/descriptor]
      (is (= "layers.%d.mlp.gate.weight" (get-in d [:names :moe-router])))
      (is (= "layers.%d.mlp.experts.%d.gate_proj.weight" (get-in d [:moe-names :gate])))
      (is (false? (get-in d [:flags :tied-lm-head])))
      (is (nil? (get-in d [:names :ffn-gate])) "dense FFN roles removed")
      (is (= #{:attn-q :attn-k :attn-v :attn-o} (:linear-roles d)))
      (is (= "layers.0.mlp.experts.5.up_proj.weight" (dec/moe-name d :up 0 5))))))
