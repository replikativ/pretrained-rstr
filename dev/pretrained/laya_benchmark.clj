(ns pretrained.laya-benchmark
  "Integration runner for the pinned typed-decisions cohort.

  Add ../finetune-rstr/src and ../finetune-rstr/test to the development
  classpath; the production pretrained-rstr artifact remains independent."
  (:require [clojure.data.json :as json]
            [clojure.java.io :as io]
            [finetune.decision.adapters.laya :as adapter]
            [finetune.decision.benchmark :as benchmark]
            [pretrained.decision.laya :as laya]))

(defn run-frozen!
  "Load a local Laya checkpoint and evaluate the pinned 8-case/40-question
  cohort. If `output` is supplied, write the complete scored artifact as JSON."
  ([model-dir] (run-frozen! model-dir nil))
  ([model-dir output]
   (let [cohort (benchmark/frozen-cohort)
         load-start (System/nanoTime)
         agent (laya/load-agent model-dir)
         load-ms (/ (- (System/nanoTime) load-start) 1.0e6)
         result (benchmark/evaluate-backend
                 (:records cohort)
                 agent
                 adapter/prediction
                 {:implementation :pretrained-rstr
                  :checkpoint :convaiinnovations/laya-typed-decisions
                  :mode :specialist :device :cpu
                  :model-dir model-dir :load-ms load-ms}
                 {:on-case (fn [{:keys [index total case-id elapsed-ms]}]
                             (println (format "[%d/%d] %s: %.1f ms"
                                              (inc index) total case-id elapsed-ms))
                             (flush))})]
     (when output
       (with-open [writer (io/writer output)]
         (json/write result writer)))
     result)))

(defn read-result [path]
  (with-open [reader (io/reader path)]
    (json/read reader :key-fn keyword)))

(defn compare-results
  "Compare canonical probabilities in two scored artifacts. Returns structural
  parity and the maximum absolute probability error."
  [left right]
  (let [lc (:cases left) rc (:cases right)
        structure-left (mapv #(select-keys % [:case-id :question-ids]) lc)
        structure-right (mapv #(select-keys % [:case-id :question-ids]) rc)
        lp (mapcat #(mapcat identity (:predictions %)) lc)
        rp (mapcat #(mapcat identity (:predictions %)) rc)]
    {:same-structure? (= structure-left structure-right)
     :probabilities (count lp)
     :max-absolute-error (if (= (count lp) (count rp))
                           (reduce max 0.0 (map #(Math/abs (- (double %1) (double %2))) lp rp))
                           Double/POSITIVE_INFINITY)}))
