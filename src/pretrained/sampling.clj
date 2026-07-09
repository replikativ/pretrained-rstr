(ns pretrained.sampling
  "Token sampling from a logits row: greedy / temperature / top-k / top-p (nucleus).
  A sampler is a fn `[^floats logits vocab] -> token-id`; the decode loop calls it
  per step. `make-sampler` builds one from a config map; temperature<=0 (or empty
  config) is greedy argmax. Pipeline (llama.cpp/HF order): take top-k candidates,
  temperature-scale + softmax, cut at the top-p nucleus, multinomial sample.

  This ns is sampling POLICY (which strategy, the RNG, the top-k/top-p pipeline). The
  greedy KERNEL is the existing raster.arrays/argmax deftm (compilable to JVM/C/GPU —
  runs where the logits live, so on the GPU profile it stays on-device)."
  (:require [raster.arrays :as ra])
  (:import [java.util Random]))

(defn- top-k-ids
  "int[] of the k largest-logit indices, sorted DESCENDING by logit. O(vocab) for the
  common case (one compare per non-candidate), via bounded insertion into a size-k array."
  ^ints [^floats lg vocab k]
  (let [k (int (min (int k) (int vocab)))
        ids (int-array k)
        vals (double-array k)]
    (java.util.Arrays/fill vals Double/NEGATIVE_INFINITY)
    (dotimes [i (int vocab)]
      (let [v (double (aget lg i))]
        (when (> v (aget vals (dec k)))
          (loop [j (dec k)]
            (if (and (> j 0) (< (aget vals (dec j)) v))
              (do (aset vals j (aget vals (dec j))) (aset ids j (aget ids (dec j))) (recur (dec j)))
              (do (aset vals j v) (aset ids j i)))))))
    ids))

(defn make-sampler
  "Build a sampler fn from {:temperature :top-k :top-p :seed}. temperature<=0 -> greedy."
  [{:keys [temperature top-k top-p seed] :as opts}]
  (let [temp (double (or temperature 1.0))]
    ;; No config, or temperature<=0 -> greedy (a vocab-wide argmax, not a full-vocab
    ;; top-k selection). This is the DEFAULT and must stay O(vocab).
    (if (or (empty? opts) (<= temp 0.0))
      (fn [^floats lg _vocab] (ra/argmax lg))   ;; greedy = raster deftm (GPU-able)
      ;; default top-k = 40 (NOT 0): top-k-ids is O(vocab*k), so k=vocab would be
      ;; O(vocab^2) — a full-vocab nucleus needs a different (sort-based) path.
      (let [k (int (or top-k 40))
            p (double (or top-p 1.0))
            ^Random rng (if seed (Random. (long seed)) (Random.))]
        (fn [^floats lg vocab]
          (let [kk (min k (int vocab))
                ids (top-k-ids lg vocab kk)
                ;; temperature-scaled softmax over the kk candidates (ids[0] = max)
                mx (/ (double (aget lg (aget ids 0))) temp)
                probs (double-array kk)
                _ (loop [j 0] (when (< j kk)
                                (aset probs j (Math/exp (- (/ (double (aget lg (aget ids j))) temp) mx)))
                                (recur (inc j))))
                psum (areduce probs j s 0.0 (+ s (aget probs j)))
                ;; top-p nucleus cut over the descending candidates
                cut (loop [j 0 c 0.0]
                      (if (< j kk)
                        (let [c2 (+ c (/ (aget probs j) psum))]
                          (if (>= c2 p) (inc j) (recur (inc j) c2)))
                        kk))
                tot (loop [j 0 s 0.0] (if (< j cut) (recur (inc j) (+ s (aget probs j))) s))
                r (* (.nextDouble rng) tot)]
            (loop [j 0 acc 0.0]
              (let [acc2 (+ acc (aget probs j))]
                (if (or (>= acc2 r) (= j (dec cut))) (aget ids j) (recur (inc j) acc2))))))))))

(def greedy (make-sampler {}))
