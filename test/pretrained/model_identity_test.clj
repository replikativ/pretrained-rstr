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

(deftest structured-execution-variants
  (let [directory (Files/createTempDirectory
                   "pretrained-model-identity-" (make-array FileAttribute 0))
        weights (.resolve directory "model.safetensors")
        variant {:name :gpu-paged
                 :cache {:attention-state {:dtype :float16 :quantization :none}}
                 :numerical {:mode :fp16-kv :determinism :reproducible-order}}]
    (try
      (Files/write weights (byte-array [1 2 3])
                   (make-array java.nio.file.OpenOption 0))
      (let [model (fixture-model directory)
            fingerprint #(model-identity/compatibility-fingerprint
                          model {:execution-variant %})]
        (testing "keyword variants hash exactly as before"
          (is (= (fingerprint :default) (model-identity/compatibility-fingerprint model))))
        (testing "cache precision is part of identity"
          (is (not= (fingerprint variant)
                    (fingerprint (assoc-in variant [:cache :attention-state :dtype]
                                           :float32)))))
        (testing "the numerical contract is readable from the variant"
          (is (= {:mode :fp16-kv :determinism :reproducible-order}
                 (model-identity/numerical-contract variant)))
          (is (nil? (model-identity/numerical-contract :gpu-q4k-paged))))
        (testing "malformed variants are rejected before hashing"
          (is (thrown-with-msg? clojure.lang.ExceptionInfo #"known :determinism"
                                (fingerprint (assoc-in variant [:numerical :determinism]
                                                       :sometimes))))
          (is (thrown-with-msg? clojure.lang.ExceptionInfo #"unsupported keys"
                                (fingerprint (assoc variant :rope :yarn)))))
        (testing "the cache names exactly the layout's groups"
          (is (thrown-with-msg? clojure.lang.ExceptionInfo #"every attention-state group"
                                (fingerprint (assoc variant :cache
                                                    {:global {:dtype :float16
                                                              :quantization :none}})))))
        (testing "the cache dtype must be the storage dtype"
          (is (= variant (model-identity/require-cache-dtype! variant :float16)))
          (is (thrown-with-msg? clojure.lang.ExceptionInfo #"storage dtype"
                                (model-identity/require-cache-dtype! variant :float32)))
          (is (= :default (model-identity/require-cache-dtype! :default :float32)))))
      (finally
        (Files/deleteIfExists weights)
        (Files/deleteIfExists directory)))))
