(ns pretrained.continuation.chunk-store
  "Immutable KV chunks in a local Boring-backed Konserve filestore.

  Every value has one contiguous `:chunk/payload` float array. Konserve 0.9.377
  can expose that nested RFC 8746 payload as a scoped read-only MemorySegment,
  avoiding decode and heap copies on the SSD-to-GPU path."
  (:require [boring.core]
            [clojure.string :as str]
            [hasch.core :as hasch]
            [konserve.core :as k]
            [konserve.filestore :refer [connect-fs-store]]
            [konserve.mmap :as kmm])
  (:import [java.lang AutoCloseable]
           [java.util UUID]))

(defn content-id
  "Return the Hasch identity for a tensor chunk and its logical prefix.

  Hasch canonicalizes primitive float arrays across platforms. The domain tag
  prevents an equal EDN value used by another object type from aliasing this
  store-ref identity."
  [chunk]
  (let [fingerprint (:chunk/model-fingerprint chunk)]
    (when-not (and (string? fingerprint) (not (str/blank? fingerprint)))
      (throw (ex-info "A durable attention-state chunk requires a model fingerprint"
                      {:model-fingerprint fingerprint})))
    (when-not (instance? UUID (:chunk/prefix-hash chunk))
      (throw (ex-info "A durable attention-state chunk requires a prefix hash"
                      {:prefix-hash (:chunk/prefix-hash chunk)})))
    (hasch/uuid [::attention-chunk-v3 chunk])))

(defn open-store
  "Open a synchronous, uncompressed Boring filestore at `directory`."
  [directory]
  (connect-fs-store (str directory)
                    :config {:encoding {:serializer :BoringSerializer}}
                    :opts {:sync? true}))

(defn describe
  "Return `{:store-key :path :bytes}` for an mmap-compatible stored chunk."
  [store store-key]
  (let [[path _] (kmm/value-location store store-key)]
    {:store-key store-key
     :path path
     :bytes (.length (java.io.File. ^String path))}))

(defn put!
  "Durably store one immutable chunk and return its catalog storage fields.

  Identical content is not rewritten. Returns `{:store-key :bytes :path}`."
  [store chunk]
  (let [store-key (content-id chunk)]
    (when-not (k/exists? store store-key {:sync? true})
      (k/assoc store store-key chunk {:immutable? true} {:sync? true}))
    (describe store store-key)))

(defn stored?
  "Return true when `store-key` is present in this chunk store."
  [store store-key]
  (k/exists? store store-key {:sync? true}))

(defn read-chunk
  "Decode and return a stored chunk, or `not-found` when it is absent.

  This is intended for background replication and inspection. GPU restoration
  should use `with-mmap-payload`, which avoids decoding the tensor payload."
  ([store store-key]
   (read-chunk store store-key nil))
  ([store store-key not-found]
   (k/get store store-key not-found {:sync? true})))

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
