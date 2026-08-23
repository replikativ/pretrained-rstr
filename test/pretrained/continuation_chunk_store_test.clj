(ns pretrained.continuation-chunk-store-test
  (:require [clojure.test :refer [deftest is]]
            [pretrained.continuation.chunk-store :as chunk-store])
  (:import [java.lang.foreign MemorySegment ValueLayout]
           [java.nio ByteOrder]
           [java.nio.file Files]))

(defn- delete-directory!
  [directory]
  (with-open [paths (Files/list directory)]
    (doseq [path (iterator-seq (.iterator paths))]
      (Files/deleteIfExists path)))
  (Files/deleteIfExists directory))

(deftest stores-and-maps-one-contiguous-fp32-payload
  (let [directory (Files/createTempDirectory
                   "pretrained-kv-chunks-"
                   (make-array java.nio.file.attribute.FileAttribute 0))
        store (chunk-store/open-store directory)
        chunk {:chunk/version 1
               :chunk/model-fingerprint "fixture-v1"
               :chunk/layout {:n-layers 1 :n-kv 1 :head-dim 2}
               :chunk/start 0
               :chunk/token-count 2
               :chunk/prefix-hash (random-uuid)
               :chunk/payload (float-array [1 2 3 4 5 6 7 8])}]
    (try
      (let [first-write (chunk-store/put! store chunk)
            second-write (chunk-store/put! store chunk)]
        (is (= (:store-key first-write) (:store-key second-write)))
        (is (= (:bytes first-write) (:bytes second-write)))
        (chunk-store/with-mmap-payload
         store (:store-key first-write)
         (fn [payload]
           (let [^MemorySegment segment (:segment payload)
                 float-le (.withOrder ValueLayout/JAVA_FLOAT_UNALIGNED
                                      ByteOrder/LITTLE_ENDIAN)]
             (is (= :float32 (:element-type payload)))
             (is (= 8 (:element-count payload)))
             (is (= [1.0 2.0 3.0 4.0 5.0 6.0 7.0 8.0]
                    (mapv #(.getAtIndex segment float-le %) (range 8))))))))
      (finally
        (delete-directory! directory)))))
