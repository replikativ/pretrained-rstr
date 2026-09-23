(ns pretrained.laya-gpu-test
  (:require [clojure.test :refer [deftest is]]
            [pretrained.arch.modernbert :as mb]
            [pretrained.decision.laya :as laya]
            [pretrained.decision.laya-gpu :as laya-gpu]))

(deftest segment-score-mask-is-block-diagonal
  (let [scores (float-array (range 32))
        segments (float-array [0 0 1 1])
        masked (mb/mask-segment-scores scores segments 4 2)]
    (is (= 32 (alength masked)))
    (doseq [i (range 4) h (range 2) j (range 4)]
      (let [idx (+ (* i 8) (* h 4) j)]
        (if (= (aget segments i) (aget segments j))
          (is (= (aget scores idx) (aget masked idx)))
          (is (< (double (aget masked idx)) -1.0e29)))))))

(deftest packed-request-keeps-question-boundaries-and-padding-separate
  (let [weights {"type_emb.weight" {:data (float-array [1 2 3 4 5 6])}}
        cpu (laya/map->LayaAgent
             {:encoder {:d-model 2 :weights weights}
              :tokenizer {:tok {:pad-id 9}}})
        agent (laya-gpu/gpu-agent cpu {:shape-bucket 4 :max-packed-tokens 16})
        items [{:ids [1 2] :question-type 0}
               {:ids [3] :question-type 2}]
        packed (#'laya-gpu/pack-items agent items)]
    (is (= [1 2 3 9] (:ids packed)))
    (is (= [0 2] (:offsets packed)))
    (is (= [2 1] (:lengths packed)))
    (is (= [0.0 0.0 1.0 2.0] (vec (:segments packed))))
    (is (= [1.0 2.0 1.0 2.0 5.0 6.0 0.0 0.0]
           (vec (:type-values packed))))))

(deftest packed-groups-respect-token-limit
  (let [items [{:id :a :ids [1 2 3]}
               {:id :b :ids [4 5]}
               {:id :c :ids [6 7 8]}]]
    (is (= [[:a :b] [:c]]
           (mapv #(mapv :id %) (#'laya-gpu/packed-groups items 5))))
    (is (thrown? clojure.lang.ExceptionInfo
                 (#'laya-gpu/packed-groups items 2)))))

(deftest gpu-shape-limits-must-fit-a-bucket
  (is (thrown? clojure.lang.ExceptionInfo
               (laya-gpu/gpu-agent nil {:shape-bucket 0})))
  (is (thrown? clojure.lang.ExceptionInfo
               (laya-gpu/gpu-agent nil {:shape-bucket 8
                                        :max-packed-tokens 7})))
  (is (thrown? clojure.lang.ExceptionInfo
               (laya-gpu/gpu-agent nil {:max-cached-shapes 0}))))

(deftest public-gpu-agent-packs-reuses-and-closes
  (let [cpu (laya/map->LayaAgent
             {:encoder {:d-model 2
                        :weights {"type_emb.weight"
                                  {:data (float-array [1 2 3 4 5 6])}}}
              :tokenizer {:tok {:pad-id 9}}})
        agent (laya-gpu/gpu-agent cpu {:shape-bucket 4 :max-packed-tokens 16})
        items [{:id :a :question {:type :choice} :ids [1 2]
                :question-type 0 :markers [0 1]}
               {:id :b :question {:type :noul} :ids [3]
                :question-type 2 :markers [0 1]}]
        compiled (atom [])
        closed (atom [])]
    (with-redefs [laya/prepare-items (fn [_ _ _] items)
                  laya-gpu/compile-decision-hidden
                  (fn [_ shape _] (swap! compiled conj shape) {:shape shape})
                  laya-gpu/decision-hidden-tokens
                  (fn [_ ids _ segments]
                    (is (= [1 2 3 9] ids))
                    (is (= [0.0 0.0 1.0 2.0] (vec segments)))
                    (float-array 8))
                  laya/answer-from-hidden
                  (fn [_ question hidden _]
                    {:type (:type question) :rows (quot (alength hidden) 2)})
                  laya-gpu/close! (fn [resident] (swap! closed conj (:shape resident)))]
      (is (= {:model :laya
              :answers {:a {:type :choice :rows 2}
                        :b {:type :noul :rows 1}}
              :usage {:input-tokens 3 :output-tokens 0}}
             (agent nil nil)))
      (is (= 4 (-> @(:cache agent) :programs vals first :shape)))
      (agent nil nil)
      (is (= [4] @compiled))
      (.close ^java.io.Closeable agent)
      (is (= [4] @closed))
      (is (thrown? clojure.lang.ExceptionInfo (agent nil nil))))))

(defn- random-floats [n seed scale]
  (let [out (float-array n)
        random (java.util.Random. seed)]
    (dotimes [i n]
      (aset out i (float (* scale (- (.nextDouble random) 0.5)))))
    out))

(defn- max-error [^floats a ^floats b]
  (reduce max 0.0 (map #(Math/abs (double (- %1 %2))) a b)))

(defn- tensor [data] {:data data})

(deftest ^:anchors resident-head-preserves-cpu-laya-semantics
  (let [seq-len 2 d 64 ffn (* 4 d) heads 1 head-dim 64
        prefix "head.layers.0."
        norm1-w (float-array d 1.0)
        norm1-b (random-floats d 1 0.02)
        qkv-w (random-floats (* 3 d d) 2 0.02)
        qkv-b (random-floats (* 3 d) 3 0.02)
        out-w (random-floats (* d d) 4 0.02)
        out-b (random-floats d 5 0.02)
        norm2-w (float-array d 1.0)
        norm2-b (random-floats d 6 0.02)
        linear1-w (random-floats (* ffn d) 7 0.02)
        linear1-b (random-floats ffn 8 0.02)
        linear2-w (random-floats (* d ffn) 9 0.02)
        linear2-b (random-floats d 10 0.02)
        type-emb (random-floats (* 3 d) 11 0.02)
        weights {(str prefix "norm1.weight") (tensor norm1-w)
                 (str prefix "norm1.bias") (tensor norm1-b)
                 (str prefix "self_attn.in_proj_weight") (tensor qkv-w)
                 (str prefix "self_attn.in_proj_bias") (tensor qkv-b)
                 (str prefix "self_attn.out_proj.weight") (tensor out-w)
                 (str prefix "self_attn.out_proj.bias") (tensor out-b)
                 (str prefix "norm2.weight") (tensor norm2-w)
                 (str prefix "norm2.bias") (tensor norm2-b)
                 (str prefix "linear1.weight") (tensor linear1-w)
                 (str prefix "linear1.bias") (tensor linear1-b)
                 (str prefix "linear2.weight") (tensor linear2-w)
                 (str prefix "linear2.bias") (tensor linear2-b)
                 "type_emb.weight" (tensor type-emb)}
        agent (laya/map->LayaAgent
               {:encoder {:d-model d :weights weights} :config {}})
        x (random-floats (* seq-len d) 12 0.2)
        question-type 1
        typed (#'laya/add-question-type agent x seq-len question-type)
        expected (#'laya/head-layer agent typed seq-len 0)
        type-row (laya-gpu/question-type-embedding agent question-type)
        type-values (float-array (* seq-len d))
        _ (dotimes [row seq-len]
            (System/arraycopy type-row 0 type-values (* row d) d))
        actual (apply laya-gpu/laya-resident-first-head
                      [x type-values norm1-w norm1-b qkv-w qkv-b out-w out-b
                       norm2-w norm2-b linear1-w linear1-b linear2-w linear2-b
                       seq-len d ffn heads head-dim 1.0e-5 (/ 1.0 8.0)])]
    (is (< (max-error expected actual) 1.0e-5))))
