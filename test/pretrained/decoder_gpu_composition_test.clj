(ns pretrained.decoder-gpu-composition-test
  (:require [clojure.string :as str]
            [clojure.test :refer [deftest is]]
            [pretrained.decoder-gpu :as decoder-gpu]
            [raster.compiler.ir.kernel-abi :as kabi]
            [raster.compiler.ir.kernel-artifact :as artifact]
            [raster.compiler.ir.kernel-launch :as launch]))

(deftest resident-tail-selects-and-gathers-each-row-deterministically
  (let [nan Float/NaN
        logits (float-array [1.0 5.0 5.0 2.0
                             9.0 nan 8.0 nan])
        embedding (float-array [0.0 10.0, 1.0 11.0, 2.0 12.0, 3.0 13.0])
        tokens (int-array 2)
        next-residual (float-array 4)]
    (decoder-gpu/decode-tail! logits embedding tokens next-residual 2 4 2)
    (is (= [1 1] (vec tokens))
        "ties and multiple NaNs both select the lowest matching column")
    (is (= [1.0 11.0 1.0 11.0] (vec next-residual))
        "each selected token gathers one complete embedding row")))

(defn- step
  [phase reads write & {:keys [inout?]}]
  (let [kernel-name (str/replace (name phase) "-" "_")
        arguments (vec (concat reads [write]))
        abi (vec (concat (map #(kabi/slot % :input :float) reads)
                         [(kabi/slot write :output :float
                                     :role (when inout? :inout))]))
        executable
        (artifact/make
         {:kernel-name kernel-name
          :source (str "__kernel void " kernel-name "("
                       (str/join ", "
                                 (concat (map #(str "__global const float* " (name %)) reads)
                                         [(str "__global float* " (name write))]))
                       ") {}")
          :abi abi
          :arguments arguments
          :launch (launch/spec {:workgroup-size [1] :group-count [1]})
          :effects {:kind :map :reads reads :writes [write]}})]
    {:phase phase
     :convention :map
     :artifact executable
     :argument-specs
     (vec (concat (map (fn [symbol] {:kind :input :sym symbol}) reads)
                  [{:kind :output :sym write}]))}))

(def ^:private synthetic-layer
  {:dtype :float
   :all-params '[input kr v qr kc vc at out]
   :array-params '[input kr v qr kc vc at out]
   :scalar-params []
   :array-roles {'input :input 'v :input 'qr :input 'kc :state 'vc :state}
   :allocs []
   :steps [(step :project '[input] 'kr)
           (step :append-key '[kr] 'kc :inout? true)
           (step :append-value '[v] 'vc :inout? true)
           (step :attention '[qr kc vc] 'at)
           (step :epilogue '[input at] 'out)]
   :result-sym 'out})

(deftest routed-attention-is-selected-by-semantic-effects
  (let [{:keys [stage pre replaced post]}
        (#'decoder-gpu/layer-step-partition synthetic-layer)]
    (is (= :routed-kv-attention (:id stage)))
    (is (= #{'kc 'vc} (:state stage)))
    (is (= #{'at} (:outputs stage)))
    (is (= [:project] (mapv :phase (:steps pre))))
    (is (= [:append-key :append-value :attention]
           (mapv :phase (:steps replaced))))
    (is (= [:epilogue] (mapv :phase (:steps post))))))
