(ns examples.interpretability
  "Transformer interpretability showcase on raster: per-layer latent tap × UMAP.

  Demonstrates the `pretrained.decoder/hidden-states` tap (per-layer resid-post) +
  `umap/fit` on a TOPIC × REGISTER crossed sentence corpus. The crossing is the trick:
  register (question/imperative/declarative) is orthogonal to topic, so a layer that
  clusters by one cannot be clustering by the other. Known result (Tenney et al. 2019,
  'BERT rediscovers the classical NLP pipeline'; Alain & Bengio 2016 probing): early
  layers organize by SURFACE FORM, late layers by MEANING — a silhouette CROSSOVER.

  Also a logit-lens panel (nostalgebraist 2020): apply final-norm + lm-head to each
  layer's resid-post and watch the next-token prediction sharpen with depth.

  Run:  clojure -M:dev -m examples.interpretability
  Outputs /tmp/interpretability.edn (coords + labels + per-layer silhouettes + lens)."
  (:require [pretrained.loader :as loader]
            [pretrained.decoder :as dec]
            [umap :as umap]
            [clojure.java.io :as io]
            [clojure.edn :as edn]))

(def model-dir (str (System/getProperty "user.home") "/Development/models/gemma-3-270m-it"))

;; --- corpus: 4 topics × 3 registers × 4 = 48 sentences, [text topic register] ---
(def corpus
  [;; astronomy
   ["How far is the nearest star from Earth?" :astronomy :question]
   ["What causes a total solar eclipse?" :astronomy :question]
   ["When will the next comet be visible?" :astronomy :question]
   ["Why do the outer planets have rings?" :astronomy :question]
   ["Point the telescope toward the eastern horizon." :astronomy :imperative]
   ["Track the moon's phase over the next week." :astronomy :imperative]
   ["Calibrate the spectrometer before the observation." :astronomy :imperative]
   ["Record the star's brightness every hour." :astronomy :imperative]
   ["A supernova briefly outshines its entire galaxy." :astronomy :declarative]
   ["Light from the sun takes eight minutes to reach us." :astronomy :declarative]
   ["Black holes bend the path of nearby starlight." :astronomy :declarative]
   ["The Milky Way contains hundreds of billions of stars." :astronomy :declarative]
   ;; cooking
   ["How long should I simmer the tomato sauce?" :cooking :question]
   ["What temperature browns butter without burning it?" :cooking :question]
   ["Which flour gives the chewiest bread?" :cooking :question]
   ["How do I keep the custard from curdling?" :cooking :question]
   ["Fold the egg whites gently into the batter." :cooking :imperative]
   ["Sear the steak over high heat for two minutes." :cooking :imperative]
   ["Let the dough rest before you roll it out." :cooking :imperative]
   ["Season the broth with salt near the end." :cooking :imperative]
   ["Caramelization begins around 160 degrees Celsius." :cooking :declarative]
   ["Kneading develops the gluten that traps gas bubbles." :cooking :declarative]
   ["Resting meat lets the juices redistribute evenly." :cooking :declarative]
   ["Acid brightens the flavor of a rich sauce." :cooking :declarative]
   ;; finance
   ["What was the closing price of the index today?" :finance :question]
   ["How does inflation erode a bond's real return?" :finance :question]
   ["Which sectors outperform when rates fall?" :finance :question]
   ["When should I rebalance the portfolio?" :finance :question]
   ["Diversify the portfolio before the rate decision." :finance :imperative]
   ["Hedge the currency exposure on the export revenue." :finance :imperative]
   ["Reinvest the dividends to compound the returns." :finance :imperative]
   ["Set a stop-loss below the support level." :finance :imperative]
   ["Bond yields move inversely to bond prices." :finance :declarative]
   ["Compounding rewards patience over long horizons." :finance :declarative]
   ["Liquidity dries up during a market panic." :finance :declarative]
   ["Diversification reduces idiosyncratic risk." :finance :declarative]
   ;; medicine
   ["What are the early symptoms of dehydration?" :medicine :question]
   ["How does a vaccine train the immune system?" :medicine :question]
   ["Which nutrients support bone density?" :medicine :question]
   ["When is a fever considered dangerous?" :medicine :question]
   ["Take the antibiotic with food twice daily." :medicine :imperative]
   ["Apply pressure to the wound to stop the bleeding." :medicine :imperative]
   ["Monitor the patient's blood pressure hourly." :medicine :imperative]
   ["Rest the injured joint and elevate it." :medicine :imperative]
   ["Antibodies bind to pathogens and mark them for removal." :medicine :declarative]
   ["Chronic stress weakens the immune response." :medicine :declarative]
   ["Insulin regulates the level of glucose in the blood." :medicine :declarative]
   ["Inflammation is the body's first line of defense." :medicine :declarative]])

(defn load-model [] (loader/from-pretrained model-dir))

(defn- encode [m text] (let [{:keys [tok encode]} (:tokenizer m)] (vec (encode tok text))))

;; per-sentence hidden-states {:layers [float[d] ...] :final float[d]}
(defn latents [m] (mapv (fn [[t _ _]] (dec/hidden-states m (encode m t))) corpus))

(defn- layer-matrix ^doubles [hss L d]
  (let [n (count hss) out (double-array (* n d))]
    (dotimes [i n]
      (let [^floats row (if (= L :final) (:final (nth hss i)) (nth (:layers (nth hss i)) L))]
        (dotimes [j d] (aset out (+ (* i d) j) (double (aget row j))))))
    out))

;; cosine silhouette on the HIGH-D latents (label geometry, not the 2D projection)
(defn- silhouette [^doubles X n d labels]
  (let [row (fn [i] (let [a (double-array d)] (dotimes [j d] (aset a j (aget X (+ (* i d) j)))) a))
        norm (fn [^doubles a] (Math/sqrt (areduce a k s 0.0 (+ s (* (aget a k) (aget a k))))))
        cos-d (fn [i j] (let [^doubles a (row i) ^doubles b (row j)
                              dp (areduce a k s 0.0 (+ s (* (aget a k) (aget b k))))
                              den (* (norm a) (norm b))]
                          (- 1.0 (/ dp (if (zero? den) 1.0 den)))))
        lab (vec labels)
        mean-to (fn [i grp] (let [ds (for [j grp :when (not= i j)] (cos-d i j))]
                              (if (seq ds) (/ (reduce + ds) (count ds)) 0.0)))
        by-label (group-by lab (range n))
        sils (for [i (range n)
                   :let [same (get by-label (lab i))
                         a (mean-to i same)
                         b (reduce min (for [[l grp] by-label :when (not= l (lab i))] (mean-to i grp)))]]
               (/ (- b a) (max a b)))]
    (/ (reduce + sils) n)))

(defn- umap-2d [^doubles X n d]
  (let [{:keys [emb]} (umap/fit X n d :out-dim 2 :seed 42 :metric :cosine)]
    (mapv (fn [i] [(aget ^doubles emb (* 2 i)) (aget ^doubles emb (inc (* 2 i)))]) (range n))))

(defn run
  "Compute UMAP coords + silhouettes per layer + a logit-lens trajectory."
  [m & {:keys [layers] :or {layers [0 3 6 9 12 15 17 :final]}}]
  (let [d (:d-model m)
        hss (latents m)
        topics (mapv #(nth % 1) corpus)
        registers (mapv #(nth % 2) corpus)
        n (count corpus)
        per-layer (into {} (for [L layers
                                 :let [X (layer-matrix hss L d)]]
                             [L {:coords (umap-2d X n d)
                                 :silhouette-topic (silhouette X n d topics)
                                 :silhouette-register (silhouette X n d registers)}]))
        ;; logit-lens on one prompt (first sentence) across all layers
        {:keys [tok decode]} (:tokenizer m)
        am (fn [^floats a] (loop [i 1 mi 0 mv (aget a 0)]
                             (if (< i (alength a)) (if (> (aget a i) mv) (recur (inc i) i (aget a i)) (recur (inc i) mi mv)) mi)))
        lens-hs (dec/hidden-states m (encode m "The capital of France is"))
        lens (mapv (fn [l] (decode tok [(am (dec/logit-lens m (nth (:layers lens-hs) l)))]))
                   (range (:n-layers m)))]
    {:model :gemma-3-270m-it :d d :n n :topics topics :registers registers
     :layers per-layer :logit-lens lens}))

(defn -main [& _]
  (let [m (load-model)
        r (run m)
        out "/tmp/interpretability.edn"]
    (spit out (pr-str r))
    (println "wrote" out)
    (println "\nsilhouette by layer (topic vs register — expect a CROSSOVER):")
    (doseq [[L v] (sort-by (fn [[L _]] (if (= L :final) 999 L)) (:layers r))]
      (println (format "  layer %-6s topic=%.3f  register=%.3f" (str L)
                       (:silhouette-topic v) (:silhouette-register v))))
    (println "\nlogit-lens across depth (prompt 'The capital of France is'):")
    (println "  " (pr-str (:logit-lens r)))
    (shutdown-agents)))
