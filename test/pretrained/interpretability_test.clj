(ns pretrained.interpretability-test
  "Gated interpretability regression: per-layer latents (decode/hidden-states) of a
  topic×register crossed corpus reorganize with depth — early layers cluster by
  surface form (register), late by meaning (topic). Asserts the silhouette crossover.
  Skips cleanly when the gemma-3-270m weights are not on disk."
  (:require [clojure.test :refer [deftest is]]))

(def ^:private model-dir
  (str (or (System/getenv "PRETRAINED_MODELS")
           (str (System/getProperty "user.home") "/Development/models"))
       "/gemma-3-270m-it"))

;; tiny inline corpus: 4 topics × 3 registers × 2 = 24 sentences [text topic register]
(def ^:private corpus
  [["How far is the nearest star?" :astronomy :q] ["Why do planets have rings?" :astronomy :q]
   ["Point the telescope upward." :astronomy :i] ["Track the moon's phase." :astronomy :i]
   ["A supernova outshines its galaxy." :astronomy :d] ["Starlight bends near black holes." :astronomy :d]
   ["How long should the sauce simmer?" :cooking :q] ["Which flour makes chewy bread?" :cooking :q]
   ["Fold the egg whites gently." :cooking :i] ["Sear the steak on high heat." :cooking :i]
   ["Caramelization begins near 160 degrees." :cooking :d] ["Kneading develops the gluten." :cooking :d]
   ["What was the closing price today?" :finance :q] ["Which sectors gain when rates fall?" :finance :q]
   ["Diversify before the rate decision." :finance :i] ["Reinvest the dividends." :finance :i]
   ["Bond yields move inversely to prices." :finance :d] ["Liquidity dries up in a panic." :finance :d]
   ["What are the signs of dehydration?" :medicine :q] ["How does a vaccine work?" :medicine :q]
   ["Take the antibiotic with food." :medicine :i] ["Elevate the injured joint." :medicine :i]
   ["Antibodies mark pathogens for removal." :medicine :d] ["Insulin regulates blood glucose." :medicine :d]])

(defn- silhouette [rows labels]
  (let [n (count rows) rows (vec rows) lab (vec labels)
        nrm (fn [^floats a] (Math/sqrt (areduce a k s 0.0 (+ s (* (aget a k) (aget a k))))))
        cosd (fn [i j] (let [^floats a (rows i) ^floats b (rows j)
                             dp (areduce a k s 0.0 (+ s (* (aget a k) (aget b k))))
                             den (* (nrm a) (nrm b))]
                         (- 1.0 (/ dp (if (zero? den) 1.0 den)))))
        by (group-by lab (range n))
        mean-to (fn [i grp] (let [ds (for [j grp :when (not= i j)] (cosd i j))]
                              (if (seq ds) (/ (reduce + ds) (count ds)) 0.0)))
        sils (for [i (range n)
                   :let [a (mean-to i (get by (lab i)))
                         b (reduce min (for [[l g] by :when (not= l (lab i))] (mean-to i g)))]]
               (/ (- b a) (max a b)))]
    (/ (reduce + sils) n)))

(deftest ^:anchors representation-reorganizes-with-depth
  (if-not (.exists (java.io.File. model-dir))
    (println "SKIP interpretability crossover (gemma-3-270m not present)")
    (let [from (requiring-resolve 'pretrained.loader/from-pretrained)
          hidden (requiring-resolve 'pretrained.decoder/hidden-states)
          m (from model-dir)
          {:keys [tok encode]} (:tokenizer m)
          hss (mapv (fn [[t _ _]] (hidden m (vec (encode tok t)))) corpus)
          topics (mapv #(nth % 1) corpus)
          registers (mapv #(nth % 2) corpus)
          early (mapv #(first (:layers %)) hss)          ; layer 0 resid-post
          late  (mapv :final hss)                        ; final hidden
          st-early (silhouette early topics)  sr-early (silhouette early registers)
          st-late  (silhouette late  topics)  sr-late  (silhouette late  registers)]
      (println (format "topic: L0=%.3f final=%.3f | register: L0=%.3f final=%.3f"
                       st-early st-late sr-early sr-late))
      (is (> st-late (+ 0.05 st-early)) "late layers cluster by MEANING better than early")
      (is (> sr-early sr-late) "early layers cluster by SURFACE FORM better than late (crossover)"))))
