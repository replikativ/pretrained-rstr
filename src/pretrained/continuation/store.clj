(ns pretrained.continuation.store
  "Mmap-friendly durable storage for continuation snapshots.

  Tensor slabs stay raw native-order FP32 so Raster can transfer directly to or
  from mapped segments. A small Boring/CBOR manifest at the end describes those
  slabs, followed by a fixed footer locating the manifest."
  (:require [boring.core :as boring])
  (:import [java.io Closeable RandomAccessFile]
           [java.lang.foreign Arena MemorySegment ValueLayout]
           [java.nio ByteOrder]
           [java.nio.channels FileChannel$MapMode]
           [java.nio.charset StandardCharsets]
           [java.security MessageDigest]
           [java.util UUID]))

(def ^:private magic
  (.getBytes "RSTRKV01" StandardCharsets/US_ASCII))

(def ^:private footer-bytes 16)

(defrecord MappedSnapshot [^Arena arena ^MemorySegment segment snapshot]
  Closeable
  (close [_] (.close arena)))

(defn- source-segment
  ^MemorySegment [source]
  (if (instance? MemorySegment source)
    source
    (MemorySegment/ofArray ^floats source)))

(defn- tensor-elements
  ^long [snapshot]
  (let [{:keys [n-kv head-dim]} (:continuation/layout snapshot)
        processed (long (:continuation/processed-count snapshot))]
    (when (or (neg? processed) (not (pos? (long n-kv)))
              (not (pos? (long head-dim))))
      (throw (ex-info "Continuation tensor dimensions are invalid"
                      {:processed-count processed :n-kv n-kv :head-dim head-dim})))
    (* processed (long n-kv) (long head-dim))))

(defn- content-uuid
  [^MemorySegment segment]
  (let [digest (.digest (doto (MessageDigest/getInstance "SHA-256")
                          (.update (.asByteBuffer segment))))
        bytes (aclone ^bytes digest)]
    ;; RFC-4122 variant/version over 128 content-derived bits.
    (aset bytes 6 (unchecked-byte (bit-or 0x50 (bit-and 0x0f (aget bytes 6)))))
    (aset bytes 8 (unchecked-byte (bit-or 0x80 (bit-and 0x3f (aget bytes 8)))))
    (let [bb (java.nio.ByteBuffer/wrap bytes)]
      (UUID. (.getLong bb) (.getLong bb)))))

(defn- snapshot-slabs
  [snapshot]
  (let [n-layers (long (get-in snapshot [:continuation/layout :n-layers]))
        elements (tensor-elements snapshot)
        byte-length (* 4 elements)
        tensors (for [kind [:key :value] layer (range n-layers)] [kind layer])]
    (loop [remaining tensors offset (long (alength ^bytes magic)) slabs []]
      (if-let [[kind layer] (first remaining)]
        (recur (next remaining) (+ offset byte-length)
               (conj slabs {:kind kind :layer layer :offset offset
                            :elements elements :byte-length byte-length}))
        [slabs offset]))))

(defn write-snapshot-with!
  "Allocate a mapped snapshot and let `populate!` fill its tensor slabs.

  `snapshot` supplies metadata including the complete layout and processed count;
  it need not contain K/V tensors. `populate!` receives that metadata with K/V
  vectors of writable `MemorySegment` slices. It must fill every slice before it
  returns. The mapping is forced and content-addressed only after it succeeds.

  Returns `{:path string :bytes long :content-id uuid}`. The supplied segments
  are valid only during `populate!`."
  [snapshot path populate!]
  (let [[slabs slab-end] (snapshot-slabs snapshot)
        manifest (-> snapshot
                     (dissoc :continuation/keys :continuation/values)
                     (assoc :continuation/slabs slabs))
        ^bytes manifest-bytes (boring/encode manifest {:profile :archival})
        manifest-length (alength manifest-bytes)
        total (+ slab-end manifest-length footer-bytes)
        file (java.io.File. (str path))]
    (with-open [raf (RandomAccessFile. file "rw")]
      (.setLength raf total)
      (let [channel (.getChannel raf)
            arena (Arena/ofConfined)
            segment (.map channel FileChannel$MapMode/READ_WRITE 0 total arena)
            long-le (.withOrder ValueLayout/JAVA_LONG_UNALIGNED ByteOrder/LITTLE_ENDIAN)
            slab-segment (fn [{:keys [offset byte-length]}]
                           (.asSlice segment (long offset) (long byte-length)))
            keys (->> slabs (filter #(= :key (:kind %))) (mapv slab-segment))
            values (->> slabs (filter #(= :value (:kind %))) (mapv slab-segment))]
        (try
          (MemorySegment/copy (MemorySegment/ofArray ^bytes magic) 0 segment 0
                              (alength ^bytes magic))
          (populate! (assoc snapshot :continuation/keys keys
                            :continuation/values values))
          (MemorySegment/copy (MemorySegment/ofArray manifest-bytes) 0 segment
                              slab-end manifest-length)
          (.set segment long-le (+ slab-end manifest-length) (long manifest-length))
          (MemorySegment/copy (MemorySegment/ofArray ^bytes magic) 0 segment
                              (+ slab-end manifest-length 8) (alength ^bytes magic))
          (.force segment)
          {:path (.getAbsolutePath file) :bytes total :content-id (content-uuid segment)}
          (finally (.close arena)))))))

(defn write-snapshot!
  "Write a portable CPU/GPU snapshot to `path` and return its content identity.

  Returns `{:path string :bytes long :content-id uuid}`. Existing content at the
  exact path is replaced; callers should normally choose a content-addressed or
  temporary path and publish it in Datahike only after this function succeeds."
  [snapshot path]
  (let [sources (concat (:continuation/keys snapshot)
                        (:continuation/values snapshot))
        expected (* 4 (tensor-elements snapshot))]
    (when-not (and (= (* 2 (long (get-in snapshot [:continuation/layout :n-layers])))
                       (count sources))
                   (every? #(= expected (.byteSize (source-segment %))) sources))
      (throw (ex-info "Continuation tensors do not match their declared layout"
                      {:expected-tensors (* 2 (get-in snapshot [:continuation/layout :n-layers]))
                       :expected-bytes-per-tensor expected})))
    (write-snapshot-with!
     snapshot path
     (fn [destinations]
       (doseq [[source ^MemorySegment destination]
               (map vector sources
                    (concat (:continuation/keys destinations)
                            (:continuation/values destinations)))]
         (MemorySegment/copy (source-segment source) 0 destination 0 expected))))))

(defn- bytes-at
  ^bytes [^MemorySegment segment offset length]
  (let [out (byte-array length)]
    (MemorySegment/copy segment (long offset) (MemorySegment/ofArray out) 0 (long length))
    out))

(defn open-snapshot
  "Map a snapshot file and return a closeable `MappedSnapshot`.

  `(:snapshot mapped)` has the portable metadata plus K/V vectors of raw
  `MemorySegment` slices suitable for Raster's ranged-transfer API. The mapping
  and every slice become invalid when the returned value is closed."
  [path]
  (let [file (java.io.File. (str path))
        arena (Arena/ofShared)
        raf (RandomAccessFile. file "r")]
    (try
      (let [channel (.getChannel raf)
            size (.size channel)
            _ (when (< size (+ (* 2 (alength ^bytes magic)) 8))
                (throw (ex-info "Continuation file is truncated" {:size size})))
            segment (.map channel FileChannel$MapMode/READ_ONLY 0 size arena)
            long-le (.withOrder ValueLayout/JAVA_LONG_UNALIGNED ByteOrder/LITTLE_ENDIAN)
            _ (when-not (and (java.util.Arrays/equals ^bytes magic
                                                     (bytes-at segment 0 (alength ^bytes magic)))
                             (java.util.Arrays/equals ^bytes magic
                                                     (bytes-at segment (- size (alength ^bytes magic))
                                                               (alength ^bytes magic))))
                (throw (ex-info "Continuation file magic does not match" {:path (str path)})))
            manifest-length (.get segment long-le (- size footer-bytes))
            manifest-start (- size footer-bytes manifest-length)
            _ (when (< manifest-start (alength ^bytes magic))
                (throw (ex-info "Continuation manifest length is invalid"
                                {:size size :manifest-length manifest-length})))
            manifest (boring/decode (bytes-at segment manifest-start manifest-length))
            slabs (:continuation/slabs manifest)
            slab-segment (fn [{:keys [offset byte-length]}]
                           (.asSlice segment (long offset) (long byte-length)))
            keys (->> slabs (filter #(= :key (:kind %))) (sort-by :layer)
                      (mapv slab-segment))
            values (->> slabs (filter #(= :value (:kind %))) (sort-by :layer)
                        (mapv slab-segment))]
        (->MappedSnapshot arena segment
                          (-> manifest
                              (dissoc :continuation/slabs)
                              (assoc :continuation/keys keys :continuation/values values))))
      (catch Throwable error
        (.close arena)
        (throw error))
      (finally (.close raf)))))

(defn materialize
  "Copy a mapped snapshot's tensor slices into JVM float arrays."
  [snapshot]
  (let [copy (fn [^MemorySegment segment]
               (let [out (float-array (quot (.byteSize segment) 4))]
                 (MemorySegment/copy segment 0 (MemorySegment/ofArray out) 0 (.byteSize segment))
                 out))]
    (-> snapshot
        (update :continuation/keys #(mapv copy %))
        (update :continuation/values #(mapv copy %)))))
