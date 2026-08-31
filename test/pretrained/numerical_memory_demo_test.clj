(ns pretrained.numerical-memory-demo-test
  (:require [clojure.test :refer [deftest is]]
            [pretrained.numerical-memory-demo :as demo])
  (:import [java.nio.file Files Path]
           [java.nio.file.attribute FileAttribute]))

(defn- delete-tree!
  [^Path root]
  (when (Files/exists root (make-array java.nio.file.LinkOption 0))
    (with-open [paths (Files/walk root (make-array java.nio.file.FileVisitOption 0))]
      (doseq [path (reverse (iterator-seq (.iterator paths)))]
        (Files/deleteIfExists path)))))

(deftest model-free-numerical-memory-roundtrip
  (let [directory (Files/createTempDirectory
                   "pretrained-numerical-memory-test-"
                   (make-array FileAttribute 0))]
    (try
      (let [result (demo/run-local! directory)]
        (is (= 2 (:stored-chunks result)))
        (is (:deduplicated-on-repeat? result))
        (is (= 512 (:cached-tokens result)))
        (is (= {:element-type :float32
                :element-count 8192
                :byte-order :little-endian}
               (:mmap-payload result))))
      (finally
        (delete-tree! directory)))))
