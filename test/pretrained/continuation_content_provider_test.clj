(ns pretrained.continuation-content-provider-test
  (:require [clojure.test :refer [deftest is]]
            [pretrained.continuation :as continuation]
            [pretrained.continuation.chunk-store :as chunk-store]
            [pretrained.continuation.content-provider :as content-provider]
            [raster.runtime.numerical-content :as content])
  (:import [java.lang AutoCloseable]
           [java.lang.foreign MemorySegment]
           [java.nio.file Files Path]
           [java.util Comparator]
           [java.util.concurrent RejectedExecutionException]))

(defn- delete-directory!
  [^Path directory]
  (when (Files/exists directory (make-array java.nio.file.LinkOption 0))
    (with-open [paths (Files/walk directory (make-array java.nio.file.FileVisitOption 0))]
      (.forEach (.sorted paths (Comparator/reverseOrder))
                (reify java.util.function.Consumer
                  (accept [_ path] (Files/deleteIfExists ^Path path)))))))

(defn- fixture-chunk
  []
  {:chunk/version 3
   :chunk/model-fingerprint "fixture-v1"
   :chunk/layout
   (-> (continuation/model-layout {:n-layers 1 :n-kv 1 :head-dim 2})
       (assoc :dtype :float16 :byte-order :little-endian)
       (assoc-in [:attention-state :dtype] :float16))
   :chunk/start 0
   :chunk/token-count 2
   :chunk/prefix-hash (random-uuid)
   :chunk/payload (short-array (range 8))})

(deftest localizes-a-remote-konserve-chunk-before-opening-a-scoped-lease
  (let [local-directory (Files/createTempDirectory
                         "pretrained-content-local-"
                         (make-array java.nio.file.attribute.FileAttribute 0))
        backend-directory (Files/createTempDirectory
                           "pretrained-content-backend-"
                           (make-array java.nio.file.attribute.FileAttribute 0))
        local-store (chunk-store/open-store local-directory)
        backend-store (chunk-store/open-store backend-directory)
        stored (chunk-store/put! backend-store (fixture-chunk))
        address (content-provider/content-address (:store-key stored))
        provider (content-provider/open-provider local-store backend-store)]
    (try
      (is (false? (chunk-store/stored? local-store (:store-key stored))))
      (let [event (content/submit-localization! provider address)
            placement (content/await-storage-event! provider event)]
        (is (= :local (:tier-id placement)))
        (is (pos? (get-in placement [:attributes :bytes])))
        (is (= 16 (get-in placement [:attributes :payload-byte-size])))
        (is (= :int16
               (get-in placement [:attributes :payload-element-type])))
        (is (integer?
             (get-in placement [:attributes :payload-file-offset])))
        (is (map? (content/storage-event-measurement provider event)))
        (content/release-storage-event! provider event))
      (is (chunk-store/stored? local-store (:store-key stored)))
      (let [lease (content/open-local-content! provider address)
            segment (content/lease-segment lease)]
        (is (= 16 (.byteSize ^MemorySegment segment)))
        (is (= :int16 (get-in lease [:placement :attributes :element-type])))
        (.close ^AutoCloseable lease)
        (is (content/lease-closed? lease)))
      (finally
        (.close ^AutoCloseable provider)
        (delete-directory! local-directory)
        (delete-directory! backend-directory)))))

(deftest localization-admission-is-bounded
  (let [local-directory (Files/createTempDirectory
                         "pretrained-content-bounded-local-"
                         (make-array java.nio.file.attribute.FileAttribute 0))
        backend-directory (Files/createTempDirectory
                           "pretrained-content-bounded-backend-"
                           (make-array java.nio.file.attribute.FileAttribute 0))
        local-store (chunk-store/open-store local-directory)
        backend-store (chunk-store/open-store backend-directory)
        chunk (fixture-chunk)
        stored (chunk-store/put! backend-store chunk)
        address (content-provider/content-address (:store-key stored))
        provider (content-provider/open-provider
                  local-store backend-store {:max-concurrent-localizations 1})
        entered (promise)
        release-read (promise)]
    (try
      (with-redefs [chunk-store/read-chunk
                    (fn [& _]
                      (deliver entered true)
                      @release-read
                      chunk)]
        (let [active (content/submit-localization! provider address)]
          (is (true? (deref entered 1000 false)))
          (let [queued (content/submit-localization! provider address)]
            (is (thrown? RejectedExecutionException
                         (content/submit-localization! provider address)))
            (is (= 1 (:localization-queue-depth
                      (content-provider/stats provider))))
            (deliver release-read true)
            (content/await-storage-event! provider active)
            (content/release-storage-event! provider active)
            (content/await-storage-event! provider queued)
            (content/release-storage-event! provider queued))))
      (finally
        (deliver release-read true)
        (.close ^AutoCloseable provider)
        (delete-directory! local-directory)
        (delete-directory! backend-directory)))))
