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

(deftest rejects-slabs-that-are-not-sized-per-token
  (let [state-slab {:name :state :tensor-key :continuation/state
                    :buffer-prefix "ss" :count 2}]
    (testing "a fixed-size recurrent state is not silently allocated per token"
      (is (thrown-with-msg?
           clojure.lang.ExceptionInfo #"only per-token slabs are implemented"
           (attention-state/layout
            {:desc {:attention-state
                    {:slabs [(assoc state-slab :elements-per-sequence 64)]}}}))))
    (testing "a slab-level extent is rejected with its unsupported keys"
      (let [data (try (attention-state/layout
                       {:desc {:attention-state
                               {:slabs [(assoc state-slab
                                               :elements-per-token 4
                                               :extent :sequence)]}}})
                      nil
                      (catch clojure.lang.ExceptionInfo e (ex-data e)))]
        (is (= [:extent] (:unsupported data)))))
    (testing "a non-positional token axis is rejected"
      (is (thrown-with-msg?
           clojure.lang.ExceptionInfo #":position token axis"
           (attention-state/layout
            {:n-layers 1 :n-kv 1 :head-dim 2
             :desc {:attention-state {:token-axis :sequence}}}))))))

(deftest one-window-predicate-for-every-decoder
  (let [gemma {:n-layers 18
               :desc {:flags {:global-layer-pattern 6 :sliding-window {:size 512}}}}]
    (is (= [5 11 17] (filterv #(attention-state/global-layer? gemma %) (range 18))))
    (is (nil? (attention-state/layer-window gemma 5)))
    (is (= 512 (attention-state/layer-window gemma 0)))
    (is (= 512 (attention-state/min-window gemma)))
    (is (nil? (attention-state/min-window {:n-layers 2 :desc {:flags {}}}))
        "a model without a window has no limit")
    (is (thrown-with-msg? clojure.lang.ExceptionInfo #"at least one token"
                          (attention-state/layer-window
                           (assoc-in gemma [:desc :flags :sliding-window :size] 0) 0)))))
