(ns pretrained.stream-cache-test
  "Model-free: the quantized-stream disk cache round-trips exactly (type-tagged
  byte[]/int[]/float[] entries, validity keyed on source size+mtime)."
  (:require [clojure.test :refer [deftest is testing]]
            [pretrained.decoder :as dec])
  (:import [java.io File]))

(deftest cache-roundtrip-and-validity
  (let [dir (doto (File/createTempFile "qstream" "") (.delete) (.mkdirs))
        src (File. dir "model.safetensors")
        _ (spit src "fake-weights")
        cf (File. dir ".qstream-q8.bin")
        w {"layers.0.a" {:wqi (int-array (range 64)) :wsi (float-array (map float (range 8)))
                         :in 32 :out 8}
           "layers.0.b" {:wqi (byte-array (range 32)) :wsi (float-array [1.5 -2.5])
                         :in 32 :out 1}}]
    (testing "invalid before write"
      (is (not (dec/stream-cache-valid? cf src))))
    (#'dec/write-stream-cache! cf src w)
    (testing "valid + exact round-trip"
      (is (dec/stream-cache-valid? cf src))
      (let [r (#'dec/read-stream-cache cf)]
        (is (= (set (keys w)) (set (keys r))))
        (is (= (vec (:wqi (r "layers.0.a"))) (vec (range 64))))
        (is (instance? (Class/forName "[B") (:wqi (r "layers.0.b"))))
        (is (= [1.5 -2.5] (mapv double (:wsi (r "layers.0.b")))))
        (is (= 32 (:in (r "layers.0.a")))) (is (= 8 (:out (r "layers.0.a"))))))
    (testing "source change invalidates"
      (spit src "fake-weights-changed!")
      (is (not (dec/stream-cache-valid? cf src))))
    (doseq [f (.listFiles dir)] (.delete ^File f))
    (.delete dir)))
