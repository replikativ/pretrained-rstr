(ns pretrained.continuation-test
  (:require [clojure.test :refer [deftest is testing]]
            [pretrained.continuation :as continuation]
            [pretrained.decoder :as decoder]))

(def ^:private model
  {:n-layers 2 :n-kv 1 :head-dim 2 :vocab 17})

(defn- fake-decode-step
  [model token position keys values & _]
  (let [row-size (* (:n-kv model) (:head-dim model))
        offset (* position row-size)]
    (doseq [layer (range (:n-layers model))
            i (range row-size)]
      (aset ^floats (nth keys layer) (+ offset i)
            (float (+ 1000 (* layer 100) (* token 10) i)))
      (aset ^floats (nth values layer) (+ offset i)
            (float (+ 2000 (* layer 100) (* token 10) i))))
    (float-array [(float token)])))

(defn- fake-logits
  [model ^floats hidden]
  (let [out (float-array (:vocab model))
        token (mod (inc (long (aget hidden 0))) (:vocab model))]
    (aset out token 1.0)
    out))

(defmacro with-fake-model
  [& body]
  `(with-redefs [decoder/decode-step fake-decode-step
                 decoder/lm-logits fake-logits]
     ~@body))

(deftest continuation-boundary-is-cache-prefix-plus-pending-token
  (with-fake-model
    (let [state (continuation/start-cpu model [2 3 4]
                                        {:max-position 12
                                         :model-fingerprint "fixture-v1"})]
      (is (= 2 (:continuation/processed-count state)))
      (is (= 4 (:continuation/pending-token state)))
      (is (= [2 3 4] (:continuation/tokens state)))
      (let [[next-state token] (continuation/step-cpu state)]
        (is (= 5 token))
        (is (= 3 (:continuation/processed-count next-state)))
        (is (= 5 (:continuation/pending-token next-state)))
        (is (= [2 3 4 5] (:continuation/tokens next-state)))))))

(deftest exported-and-restored-generation-matches-uninterrupted-generation
  (with-fake-model
    (let [initial (continuation/start-cpu model [2 3 4]
                                          {:max-position 12
                                           :model-fingerprint "fixture-v1"})
          uninterrupted (continuation/advance-cpu initial 6)
          split-start (continuation/start-cpu model [2 3 4]
                                              {:max-position 12
                                               :model-fingerprint "fixture-v1"})
          first-part (continuation/advance-cpu split-start 2)
          snapshot (continuation/export-cpu (:continuation first-part))
          restored (continuation/restore-cpu model snapshot
                                             {:max-position 12
                                              :model-fingerprint "fixture-v1"})
          second-part (continuation/advance-cpu restored 4)]
      (is (= (:tokens uninterrupted)
             (into (:tokens first-part) (:tokens second-part))))
      (is (= (:continuation/tokens (:continuation uninterrupted))
             (:continuation/tokens (:continuation second-part))))
      (testing "only occupied rows are exported"
        (is (= (* (:continuation/processed-count snapshot) 2)
               (alength ^floats (first (:continuation/keys snapshot))))))
      (testing "restored arrays regain the requested runtime capacity"
        (is (= (* 12 2)
               (alength ^floats (first (:continuation/keys restored)))))))))

(deftest restore-rejects-wrong-model-identity-and-capacity
  (with-fake-model
    (let [state (continuation/start-cpu model [1 2 3]
                                        {:max-position 8
                                         :model-fingerprint "fixture-v1"})
          snapshot (continuation/export-cpu state)]
      (is (thrown-with-msg? clojure.lang.ExceptionInfo #"fingerprint"
                            (continuation/restore-cpu model snapshot
                                                      {:max-position 8
                                                       :model-fingerprint "other"})))
      (is (thrown-with-msg? clojure.lang.ExceptionInfo #"does not fit"
                            (continuation/restore-cpu model snapshot
                                                      {:max-position 1
                                                       :model-fingerprint "fixture-v1"}))))))

(deftest empty-prompts-and-capacity-overruns-fail-loudly
  (with-fake-model
    (is (thrown-with-msg? clojure.lang.ExceptionInfo #"non-empty"
                          (continuation/start-cpu model [] {:max-position 4})))
    (let [state (continuation/start-cpu model [1 2] {:max-position 2})
          full (:continuation (continuation/advance-cpu state 1))]
      (is (thrown-with-msg? clojure.lang.ExceptionInfo #"maximum position"
                            (continuation/step-cpu full))))))
