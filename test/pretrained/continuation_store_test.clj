(ns pretrained.continuation-store-test
  (:require [clojure.test :refer [deftest is testing]]
            [pretrained.continuation :as continuation]
            [pretrained.continuation.store :as store])
  (:import [java.lang.foreign MemorySegment]))

(def ^:private layout
  (continuation/model-layout {:n-layers 1 :n-kv 1 :head-dim 2}))

(deftest mmap-snapshot-roundtrip
  (let [file (java.io.File/createTempFile "pretrained-kv-" ".rstrkv")
        snapshot {:continuation/version 1
                  :continuation/model-fingerprint "fixture-v1"
                  :continuation/layout layout
                  :continuation/processed-count 2
                  :continuation/pending-token 7
                  :continuation/tokens [1 2 7]
                  :continuation/keys [(float-array [1 2 3 4])]
                  :continuation/values [(float-array [5 6 7 8])]}]
    (try
      (let [written (store/write-snapshot! snapshot file)]
        (is (uuid? (:content-id written)))
        (is (= (.length file) (:bytes written)))
        (with-open [mapped (store/open-snapshot file)]
          (let [mapped-snapshot (:snapshot mapped)
                materialized (store/materialize mapped-snapshot)]
            (testing "metadata is ordinary CBOR and tensors remain mapped slabs"
              (is (= 2 (:continuation/processed-count mapped-snapshot)))
              (is (instance? java.lang.foreign.MemorySegment
                             (first (:continuation/keys mapped-snapshot)))))
            (testing "materialization is an explicit CPU-only operation"
              (is (= [1.0 2.0 3.0 4.0]
                     (vec (first (:continuation/keys materialized)))))
              (is (= [5.0 6.0 7.0 8.0]
                     (vec (first (:continuation/values materialized)))))))))
      (finally
        (.delete file)))))

(deftest direct-mapped-snapshot-population
  (let [file (java.io.File/createTempFile "pretrained-kv-direct-" ".rstrkv")
        metadata {:continuation/version 1
                  :continuation/model-fingerprint "fixture-v1"
                  :continuation/layout layout
                  :continuation/processed-count 2
                  :continuation/pending-token 7
                  :continuation/tokens [1 2 7]}]
    (try
      (store/write-snapshot-with!
       metadata file
       (fn [destinations]
         (is (every? #(instance? MemorySegment %)
                     (concat (:continuation/keys destinations)
                             (:continuation/values destinations))))
         (MemorySegment/copy (MemorySegment/ofArray (float-array [1 2 3 4])) 0
                             (first (:continuation/keys destinations)) 0 16)
         (MemorySegment/copy (MemorySegment/ofArray (float-array [5 6 7 8])) 0
                             (first (:continuation/values destinations)) 0 16)))
      (with-open [mapped (store/open-snapshot file)]
        (let [snapshot (store/materialize (:snapshot mapped))]
          (is (= [1.0 2.0 3.0 4.0]
                 (vec (first (:continuation/keys snapshot)))))
          (is (= [5.0 6.0 7.0 8.0]
                 (vec (first (:continuation/values snapshot)))))))
      (finally
        (.delete file)))))
