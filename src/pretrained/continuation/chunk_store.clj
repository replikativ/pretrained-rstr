(ns pretrained.continuation.chunk-store
  "Immutable KV chunks in a local Boring-backed Konserve filestore.

  Every value has one contiguous `:chunk/payload` float array. Konserve 0.9.379
  can expose that nested RFC 8746 payload as a scoped read-only MemorySegment,
  avoiding decode and heap copies on the SSD-to-GPU path."
  (:require [boring.core]
            [clojure.core.async :refer [<!!]]
            [clojure.string :as str]
            [hasch.core :as hasch]
            [konserve.core :as k]
            [konserve.filestore :refer [connect-fs-store]]
            [konserve.mmap :as kmm]
            [konserve.tiered :as tiered])
  (:import [java.lang AutoCloseable]
           [java.util UUID]))

(def ^:private durability-receipt-key ::durability-receipt)

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

(defn put-write-behind!
  "Store one immutable chunk through a write-behind tiered store.

  `local-store` must be the tiered store's mmap-compatible frontend. The
  returned map describes that local copy and carries an internal durability
  receipt for `await-durable!`. This function returns after the local write;
  the authoritative backend write remains asynchronous.

  The tiered write is deliberately repeated when the local content-addressed
  object already exists. A previous backend attempt may have failed, so local
  presence alone cannot prove global durability."
  [local-store tiered-store chunk]
  (let [store-key (content-id chunk)
        {:keys [opts receipt]}
        (tiered/with-write-behind-receipt {:sync? true})]
    (k/assoc tiered-store store-key chunk {:immutable? true} opts)
    (assoc (describe local-store store-key) durability-receipt-key receipt)))

(defn local-result
  "Return stored chunk fields without its private durability receipt."
  [stored]
  (dissoc stored durability-receipt-key))

(defn await-durable!
  "Wait for a stored chunk's authoritative backend write.

  Plain local `put!` results have no receipt and return immediately. A failed
  write-behind receipt throws with the storage identity in its exception data.
  Returns the public local storage fields."
  [stored]
  (if-let [receipt (get stored durability-receipt-key)]
    (let [{:keys [status error] :as outcome} (<!! receipt)]
      (case status
        :succeeded (local-result stored)
        :failed (throw (ex-info "The authoritative continuation chunk write failed"
                                {:store-key (:store-key stored)
                                 :outcome (dissoc outcome :error)}
                                error))
        (throw (ex-info "The continuation chunk durability receipt was invalid"
                        {:store-key (:store-key stored)
                         :outcome outcome}))))
    stored))

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
