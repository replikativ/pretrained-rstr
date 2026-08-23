(ns pretrained.continuation.chunk-store
  "Immutable KV chunks in a local Boring-backed Konserve filestore.

  Every value has one contiguous `:chunk/payload` float array. Konserve 0.9.377
  can expose that nested RFC 8746 payload as a scoped read-only MemorySegment,
  avoiding decode and heap copies on the SSD-to-GPU path."
  (:require [boring.core :as boring]
            [clojure.string :as str]
            [konserve.core :as k]
            [konserve.filestore :refer [connect-fs-store]]
            [konserve.mmap :as kmm])
  (:import [java.lang AutoCloseable]
           [java.nio ByteBuffer ByteOrder]
           [java.nio.charset StandardCharsets]
           [java.security MessageDigest]
           [java.util UUID]))

(def ^:private hash-domain
  (.getBytes "pretrained-rstr/attention-chunk-content/v2" StandardCharsets/UTF_8))

(defn- update-long!
  [^MessageDigest digest value]
  (let [buffer (ByteBuffer/allocate Long/BYTES)]
    (.putLong buffer (long value))
    (.update digest (.array buffer))))

(defn- digest-uuid
  [^bytes digest]
  (let [bytes (aclone digest)]
    (aset bytes 6 (unchecked-byte (bit-or 0x50 (bit-and 0x0f (aget bytes 6)))))
    (aset bytes 8 (unchecked-byte (bit-or 0x80 (bit-and 0x3f (aget bytes 8)))))
    (let [buffer (ByteBuffer/wrap bytes)]
      (UUID. (.getLong buffer) (.getLong buffer)))))

(defn content-id
  "Return a deterministic identity for a tensor chunk and its logical prefix."
  [chunk]
  (let [digest (MessageDigest/getInstance "SHA-256")
        prefix ^UUID (:chunk/prefix-hash chunk)
        fingerprint-value (:chunk/model-fingerprint chunk)
        _ (when-not (and (string? fingerprint-value)
                         (not (str/blank? fingerprint-value)))
            (throw (ex-info "A durable attention-state chunk requires a model fingerprint"
                            {:model-fingerprint fingerprint-value})))
        fingerprint (.getBytes ^String fingerprint-value
                               StandardCharsets/UTF_8)
        payload ^floats (:chunk/payload chunk)
        payload-bytes (doto (ByteBuffer/allocate (* Float/BYTES (alength payload)))
                        (.order (ByteOrder/nativeOrder)))]
    (.update digest hash-domain)
    (update-long! digest (.getMostSignificantBits prefix))
    (update-long! digest (.getLeastSignificantBits prefix))
    (update-long! digest (alength fingerprint))
    (.update digest fingerprint)
    (let [attention-layout (get-in chunk [:chunk/layout :attention-state])
          ^bytes layout-bytes (boring/encode attention-layout {:profile :archival})]
      (update-long! digest (alength layout-bytes))
      (.update digest layout-bytes))
    (update-long! digest (:chunk/start chunk))
    (update-long! digest (:chunk/token-count chunk))
    ;; A float[] heap segment cannot expose ByteBuffer on JDK 25. FloatBuffer.put
    ;; performs the conversion as one bulk operation without a per-value Clojure loop.
    (.put (.asFloatBuffer payload-bytes) payload)
    (.update digest payload-bytes)
    (digest-uuid (.digest digest))))

(defn open-store
  "Open a synchronous, uncompressed Boring filestore at `directory`."
  [directory]
  (connect-fs-store (str directory)
                    :config {:encoding {:serializer :BoringSerializer}}
                    :opts {:sync? true}))

(defn put!
  "Durably store one immutable chunk and return its catalog storage fields.

  Identical content is not rewritten. Returns `{:store-key :bytes :path}`."
  [store chunk]
  (let [store-key (content-id chunk)]
    (when-not (k/exists? store store-key {:sync? true})
      (k/assoc store store-key chunk {:immutable? true} {:sync? true}))
    (let [[path _] (kmm/value-location store store-key)]
      {:store-key store-key
       :path path
       :bytes (.length (java.io.File. ^String path))})))

(defn mmap-payload
  "Return `[payload arena]` for a stored chunk's no-copy FP32 payload.

  `payload` is Konserve's descriptor, including `:segment`, and is valid only
  until the caller closes `arena`. Prefer `with-mmap-payload`."
  [store store-key]
  (kmm/mmap-payload store store-key [:chunk/payload]))

(defn with-mmap-payload
  "Call `consume!` with a scoped no-copy payload descriptor.

  The mapping closes before this function returns; `consume!` must not retain the
  descriptor or its MemorySegment. Returns the callback result."
  [store store-key consume!]
  (let [[payload arena] (mmap-payload store store-key)]
    (with-open [_closeable ^AutoCloseable arena]
      (consume! payload))))
