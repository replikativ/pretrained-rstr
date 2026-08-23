(ns pretrained.attention-state-test
  (:require [clojure.test :refer [deftest is testing]]
            [pretrained.attention-state :as attention-state]))

(deftest resolves-default-kv-layout
  (let [layout (attention-state/layout
                {:n-layers 2 :n-kv 3 :head-dim 4})]
    (is (= :kv (:kind layout)))
    (is (= [[:key 2 12] [:value 2 12]]
           (mapv (juxt :name :count :elements-per-token) (:slabs layout))))
    (is (= :kc1 (attention-state/buffer-key (first (:slabs layout)) 1)))))

(deftest resolves-heterogeneous-attention-state
  (let [model {:desc {:attention-state
                      {:kind :latent
                       :slabs [{:name :latent
                                :tensor-key :continuation/latent
                                :buffer-prefix "lc"
                                :count 2
                                :elements-per-token 3}
                               {:name :rope
                                :tensor-key :continuation/rope
                                :buffer-prefix "rc"
                                :count 2
                                :elements-per-token 1}]}}}
        layout (attention-state/layout model)]
    (is (= :latent (:kind layout)))
    (is (= [0 6 12 14]
           (mapv :element-offset (attention-state/payload-plan layout 2))))
    (is (= [6 6 2 2]
           (mapv :elements (attention-state/payload-plan layout 2))))))

(deftest rejects-incomplete-and-ambiguous-layouts
  (testing "missing model dimensions report descriptor context"
    (is (thrown-with-msg? clojure.lang.ExceptionInfo
                          #"features must be positive"
                          (attention-state/layout {:n-layers 1}))))
  (testing "runtime identifiers are unique"
    (is (thrown-with-msg?
         clojure.lang.ExceptionInfo #"identifiers must be unique"
         (attention-state/layout
          {:desc {:attention-state
                  {:slabs [{:name :a :tensor-key :continuation/a
                            :buffer-prefix "x" :count 1 :elements-per-token 1}
                           {:name :b :tensor-key :continuation/b
                            :buffer-prefix "x" :count 1 :elements-per-token 1}]}}})))))
