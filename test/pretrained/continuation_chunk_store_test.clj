(ns pretrained.continuation-chunk-store-test
  (:require [clojure.test :refer [deftest is]]
            [hasch.core :as hasch]
            [pretrained.continuation :as continuation]
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

(deftest stores-and-maps-one-contiguous-fp16-carrier-payload
  (let [directory (Files/createTempDirectory
                   "pretrained-kv-chunks-"
                   (make-array java.nio.file.attribute.FileAttribute 0))
        store (chunk-store/open-store directory)
        values [1 2 3 4 5 6 7 8]
        chunk {:chunk/version 3
               :chunk/model-fingerprint "fixture-v1"
               :chunk/layout
               (-> (continuation/model-layout
                    {:n-layers 1 :n-kv 1 :head-dim 2})
                   (assoc :dtype :float16 :byte-order :little-endian)
                   (assoc-in [:attention-state :dtype] :float16))
               :chunk/start 0
               :chunk/token-count 2
               :chunk/prefix-hash (random-uuid)
               :chunk/payload
               (short-array (map #(Float/floatToFloat16 (float %)) values))}]
    (try
      (let [first-write (chunk-store/put! store chunk)
            second-write (chunk-store/put! store chunk)]
        (is (= (hasch/uuid [::chunk-store/attention-chunk-v3 chunk])
               (:store-key first-write)))
        (is (= (:store-key first-write) (:store-key second-write)))
        (is (= (:bytes first-write) (:bytes second-write)))
        (chunk-store/with-mmap-payload
         store (:store-key first-write)
         (fn [payload]
           (let [^MemorySegment segment (:segment payload)
                 short-le (.withOrder ValueLayout/JAVA_SHORT_UNALIGNED
                                      ByteOrder/LITTLE_ENDIAN)]
             (is (= :int16 (:element-type payload)))
             (is (= 8 (:element-count payload)))
             (is (= (mapv double values)
                    (mapv #(double
                            (Float/float16ToFloat
                             (.getAtIndex segment short-le %)))
                          (range 8))))))))
      (finally
        (delete-directory! directory)))))
