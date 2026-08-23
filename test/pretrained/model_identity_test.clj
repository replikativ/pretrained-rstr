(ns pretrained.model-identity-test
  (:require [clojure.test :refer [deftest is testing]]
            [pretrained.model-identity :as model-identity])
  (:import [java.nio.file Files]
           [java.nio.file.attribute FileAttribute]))

(defn- fixture-model
  [directory]
  {:dir (str directory)
   :arch "Fixture"
   :config {:hidden_size 2 :num_hidden_layers 1}
   :desc {:arch :fixture :flags {:rope :single}}
   :n-layers 1 :n-kv 1 :head-dim 2})

(deftest fingerprints-weights-layout-and-execution-variant
  (let [directory (Files/createTempDirectory
                   "pretrained-model-identity-" (make-array FileAttribute 0))
        weights (.resolve directory "model.safetensors")]
    (try
      (Files/write weights (byte-array [1 2 3])
                   (make-array java.nio.file.OpenOption 0))
      (let [model (fixture-model directory)
            first-id (model-identity/compatibility-fingerprint model)
            same-id (model-identity/compatibility-fingerprint model)]
        (is (= first-id same-id))
        (is (.startsWith first-id "sha256:"))
        (is (not= first-id
                  (model-identity/compatibility-fingerprint
                   model {:execution-variant :q8})))
        (Files/write weights (byte-array [1 2 4])
                     (make-array java.nio.file.OpenOption 0))
        (is (not= first-id
                  (model-identity/compatibility-fingerprint model))))
      (finally
        (Files/deleteIfExists weights)
        (Files/deleteIfExists directory)))))

(deftest accepts-an-immutable-external-weights-id
  (let [model (dissoc (fixture-model nil) :dir)]
    (is (= (model-identity/compatibility-fingerprint
            model {:weights-id "hf-revision:abc"})
           (model-identity/compatibility-fingerprint
            model {:weights-id "hf-revision:abc"})))
    (testing "an unidentifiable checkpoint is rejected"
      (is (thrown-with-msg? clojure.lang.ExceptionInfo #"weights-id"
                            (model-identity/compatibility-fingerprint model))))))
