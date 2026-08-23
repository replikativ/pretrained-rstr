(ns pretrained.continuation-chunk-test
  (:require [clojure.test :refer [deftest is testing]]
            [pretrained.continuation :as continuation]
            [pretrained.continuation.chunk :as chunk]))

(deftest prefix-chain-commits-to-causal-ancestry
  (let [planned (chunk/plan [1 2 3 4 5 99] 5 2)]
    (is (= [[0 2 [1 2]] [2 2 [3 4]] [4 1 [5]]]
           (mapv (juxt :chunk/start :chunk/token-count :chunk/tokens) planned)))
    (is (nil? (:chunk/parent-hash (first planned))))
    (is (= (:chunk/prefix-hash (first planned))
           (:chunk/parent-hash (second planned))))
    (is (= planned (chunk/plan [1 2 3 4 5 100] 5 2))
        "the pending token is outside the materialized KV chain")
    (testing "the same suffix under another prefix has another identity"
      (let [left (last (chunk/plan [1 2 7 8] 4 2))
            right (last (chunk/plan [3 4 7 8] 4 2))]
        (is (= (:chunk/tokens left) (:chunk/tokens right)))
        (is (not= (:chunk/prefix-hash left) (:chunk/prefix-hash right)))))))

(deftest continuation-plan-keeps-the-pending-token-separate
  (let [state {:continuation/processed-count 3
               :continuation/pending-token 4
               :continuation/tokens [1 2 3 4]}
        planned (chunk/continuation-plan state 2)]
    (is (= 3 (:processed-count planned)))
    (is (= 4 (:pending-token planned)))
    (is (= 2 (count (:chunks planned))))
    (is (= (:chunk/prefix-hash (last (:chunks planned))) (:tail-hash planned)))))

(deftest cpu-chunks-copy-only-their-token-range
  (let [model {:n-layers 1 :n-kv 1 :head-dim 2}
        state {:continuation/backend :cpu
               :continuation/model model
               :continuation/model-fingerprint "fixture-v1"
               :continuation/layout (continuation/model-layout model)
               :continuation/processed-count 4
               :continuation/pending-token 5
               :continuation/tokens [1 2 3 4 5]
               :continuation/keys [(float-array [10 11 20 21 30 31 40 41])]
               :continuation/values [(float-array [50 51 60 61 70 71 80 81])]}
        descriptor (second (chunk/plan (:continuation/tokens state) 4 2))
        tensor-chunk (chunk/cpu-tensor-chunk state descriptor)]
    (is (= [30.0 31.0 40.0 41.0 70.0 71.0 80.0 81.0]
           (vec (:chunk/payload tensor-chunk))))
    (is (= 4 (:chunk/elements-per-slab tensor-chunk)))
    (is (= 2 (:chunk/start tensor-chunk)))
    (is (= 2 (:chunk/token-count tensor-chunk)))))

(deftest cpu-chunks-follow-a-heterogeneous-slab-layout
  (let [model {:desc {:attention-state
                      {:kind :latent
                       :slabs [{:name :latent :tensor-key :continuation/latent
                                :buffer-prefix "lc" :count 1
                                :elements-per-token 2}
                               {:name :rope :tensor-key :continuation/rope
                                :buffer-prefix "rc" :count 1
                                :elements-per-token 1}]}}}
        state {:continuation/backend :cpu
               :continuation/model model
               :continuation/model-fingerprint "latent-v1"
               :continuation/layout (continuation/model-layout model)
               :continuation/processed-count 3
               :continuation/pending-token 4
               :continuation/tokens [1 2 3 4]
               :continuation/latent [(float-array [10 11 20 21 30 31])]
               :continuation/rope [(float-array [40 50 60])]}
        descriptor (second (chunk/plan (:continuation/tokens state) 3 2))
        tensor-chunk (chunk/cpu-tensor-chunk state descriptor)]
    (is (= [30.0 31.0 60.0] (vec (:chunk/payload tensor-chunk))))
    (is (= [2 1] (mapv :elements (:chunk/slabs tensor-chunk))))
    (is (nil? (:chunk/elements-per-slab tensor-chunk)))))
