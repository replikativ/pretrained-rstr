(ns pretrained.laya-cpu-benchmark
  "Controlled public-API benchmark for the shared Laya CPU reference case."
  (:require [clojure.data.json :as json]
            [clojure.java.io :as io]
            [pretrained.decision.laya :as laya]
            [raster.linalg.blas :as blas]))

(def fixture-resource "pretrained/laya_benchmark_case.json")

(defn fixture []
  (with-open [reader (io/reader (io/resource fixture-resource))]
    (json/read reader :key-fn keyword)))

(defn- percentile [sorted-values fraction]
  (nth sorted-values
       (long (Math/round (* fraction (double (dec (count sorted-values))))))))

(defn statistics [samples]
  (let [samples (mapv double samples)
        sorted-values (vec (sort samples))]
    {:samples-ms samples
     :min-ms (first sorted-values)
     :median-ms (percentile sorted-values 0.5)
     :p90-ms (percentile sorted-values 0.9)
     :max-ms (peek sorted-values)}))

(defn- timed [f]
  (let [started (System/nanoTime)
        value (f)]
    [value (/ (- (System/nanoTime) started) 1.0e6)]))

(defn run
  "Load `model-dir`, then measure first and steady public predictions.

  Set MKL_NUM_THREADS and OMP_NUM_THREADS before starting the JVM. Warmup runs
  are reported separately and never enter the steady statistics."
  [model-dir {:keys [rounds warmup] :or {rounds 7 warmup 2}}]
  (when-not (pos? (long rounds))
    (throw (ex-info "benchmark rounds must be positive" {:rounds rounds})))
  (let [{:keys [id state questions]} (fixture)
        [agent load-ms] (timed #(laya/load-agent model-dir))
        built (mapv #(laya/build-sequence (:tokenizer agent) state %) (vals questions))
        predict #(laya/predict agent state questions)
        [first-result first-ms] (timed predict)
        warmup-ms (mapv (fn [_] (second (timed predict))) (range warmup))
        steady (mapv (fn [_]
                       (let [[result elapsed] (timed predict)]
                         (when-not (= first-result result)
                           (throw (ex-info "Laya benchmark result changed between runs"
                                           {:fixture id})))
                         elapsed))
                     (range rounds))]
    {:schema :pretrained/laya-benchmark-v1
     :fixture id
     :implementation :pretrained-rstr
     :device :cpu
     :runtime {:java (System/getProperty "java.version")
               :os (System/getProperty "os.name")
               :arch (System/getProperty "os.arch")
               :processors (.availableProcessors (Runtime/getRuntime))
               :blas (blas/backend)
               :mkl-num-threads (System/getenv "MKL_NUM_THREADS")
               :omp-num-threads (System/getenv "OMP_NUM_THREADS")}
     :shape {:questions (count questions)
             :token-lengths (mapv (comp count :ids) built)
             :padded-tokens (reduce max (map (comp count :ids) built))}
     :load-ms load-ms
     :first-predict-ms first-ms
     :warmup-ms warmup-ms
     :steady-predict (statistics steady)
     :prediction first-result}))

(defn -main [& [model-dir rounds warmup]]
  (when-not model-dir
    (throw (ex-info "usage: ... -m pretrained.laya-cpu-benchmark MODEL_DIR [ROUNDS] [WARMUP]"
                    {})))
  (println (json/write-str
            (run model-dir {:rounds (if rounds (parse-long rounds) 7)
                            :warmup (if warmup (parse-long warmup) 2)})))
  (shutdown-agents))
