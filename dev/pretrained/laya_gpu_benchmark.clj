(ns pretrained.laya-gpu-benchmark
  "Controlled public-API Intel GPU benchmark for the shared Laya fixture."
  (:require [clojure.data.json :as json]
            [pretrained.decision :as decision]
            [pretrained.laya-cpu-benchmark :as cpu-bench]))

(defn- timed [f]
  (let [started (System/nanoTime)
        value (f)]
    [value (/ (- (System/nanoTime) started) 1.0e6)]))

(defn run
  [model-dir {:keys [rounds warmup target gemm-precision]
              :or {rounds 7 warmup 2 target :ze:0
                   gemm-precision :mixed-f16-f32}}]
  (when-not (pos? (long rounds))
    (throw (ex-info "benchmark rounds must be positive" {:rounds rounds})))
  (let [{:keys [id state questions]} (cpu-bench/fixture)
        [agent load-ms] (timed #(decision/load-decision
                                 :laya-english
                                 {:dir model-dir :gpu? true :target target
                                  :gemm-precision gemm-precision}))]
    (try
      (let [predict #(agent state questions)
            [first-result first-ms] (timed predict)
            warmup-ms (mapv (fn [_] (second (timed predict))) (range warmup))
            steady (mapv (fn [_]
                           (let [[result elapsed] (timed predict)]
                             (when-not (= first-result result)
                               (throw (ex-info "Laya GPU prediction changed between runs"
                                               {:fixture id})))
                             elapsed))
                         (range rounds))]
        {:schema :pretrained/laya-benchmark-v1
         :fixture id
         :implementation :pretrained-rstr
         :device target
         :runtime {:java (System/getProperty "java.version")
                   :gemm-precision gemm-precision}
         :load-ms load-ms
         :first-predict-ms first-ms
         :warmup-ms warmup-ms
         :steady-predict (cpu-bench/statistics steady)
         :prediction first-result})
      (finally (.close ^java.io.Closeable agent)))))

(defn -main [& [model-dir rounds warmup]]
  (when-not model-dir
    (throw (ex-info "usage: ... -m pretrained.laya-gpu-benchmark MODEL_DIR [ROUNDS] [WARMUP]"
                    {})))
  (println (json/write-str
            (run model-dir {:rounds (if rounds (parse-long rounds) 7)
                            :warmup (if warmup (parse-long warmup) 2)})))
  (shutdown-agents))
