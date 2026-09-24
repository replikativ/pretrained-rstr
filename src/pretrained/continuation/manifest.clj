(ns pretrained.continuation.manifest
  "Project catalogued continuation state onto Raster's NumericalStateManifest.

  The Datahike catalog stays the index for prefix lookup and placement. A
  manifest is a certified, storage-neutral view of one boundary's state: its
  identity is the catalog node of the boundary, which covers the model
  fingerprint and the prefix hash, and its parent is the previous boundary's
  node. Each retention group is one field `[members, tokens, row-width]`
  whose grid cells are the group's stored parts; a part's payload is ordered
  by member, then token rows, which is row-major for that field, so no blob is
  opened or re-encoded. A window group's field covers only the chunks the next
  token attends to and records its token origin.

  Restore reads its load list from `load-plan`, so what a boundary contains is
  decided once, here and in `pretrained.continuation.parts`."
  (:require [pretrained.attention-state :as attention-state]
            [pretrained.continuation.catalog :as catalog]
            [pretrained.continuation.parts :as parts]
            [raster.compiler.ir.abstract-value :as abstract-value]
            [raster.compiler.ir.numerical-state :as numerical-state])
  (:import [java.util UUID]))

(def ^:private raster-dtypes
  {:float32 {:dtype :float :bytes 4}
   :float16 {:dtype :half :bytes 2}})

(def ^:private legacy-content-algorithm
  "The algorithm of catalog nodes written before nodes recorded theirs."
  :hasch/attention-chunk-v3)

(defn chain
  "Return the catalogued chunk entries from the root to `prefix-hash`.

  Throws when the chain is broken, so a manifest never describes a boundary
  whose earlier state is missing."
  [database model-fingerprint prefix-hash]
  (loop [hash prefix-hash
         entries ()]
    (if (nil? hash)
      (vec entries)
      (let [entry (catalog/lookup-chunk database model-fingerprint hash)]
        (when-not entry
          (throw (ex-info "Continuation chain is missing a chunk"
                          {:model-fingerprint model-fingerprint
                           :boundary prefix-hash
                           :missing hash})))
        (recur (:kv/parent-hash entry) (conj entries entry))))))

(defn- grid-step
  "Return the chunk size of `entries`: every chunk but the last is full."
  [entries]
  (let [size (:kv/token-count (first entries))]
    (doseq [entry (butlast entries)]
      (when-not (= size (:kv/token-count entry))
        (throw (ex-info "Interior continuation chunks must share one size"
                        {:expected size :entry (:kv/prefix-hash entry)
                         :actual (:kv/token-count entry)}))))
    size))

(defn- group-width
  [attention-layout group]
  (let [by-name (into {} (map (juxt :name identity)) (:slabs attention-layout))
        widths (into #{} (map #(:elements-per-token (get by-name (first %))))
                     (:members group))]
    (when-not (= 1 (count widths))
      (throw (ex-info "A group field needs one row width"
                      {:group (:id group) :widths widths})))
    (first widths)))

(defn- group-field
  [attention-layout record {:keys [dtype bytes]} selection step group]
  (let [members (:members group)
        rows (count members)
        width (group-width attention-layout group)
        selected (for [chunk (:chunks selection)
                       part (:parts chunk)
                       :when (= (:id group) (:group part))]
                   [chunk part])
        origin (long (get (:window-floors selection) (:id group) 0))
        extent (- (long (:boundary selection)) origin)]
    (numerical-state/field
     {:id (:id group)
      :value (abstract-value/tensor
              {:dtype dtype :shape [rows extent width]
               :attributes {:extent (:extent group)}})
      :chunk-shape [rows step width]
      :coordinate-space {:token-origin origin :members members}
      :chunks
      (mapv (fn [[{:keys [entry start token-count]} part]]
              (numerical-state/chunk
               {:id (:kv/prefix-hash entry)
                :offsets [0 (- (long start) origin) 0]
                :shape [rows token-count width]
                :logical-byte-length (* (long bytes) rows (long token-count) (long width))
                :stored-byte-length (long (:bytes part))
                :content (numerical-state/content-address
                          (or (:kv/content-algorithm entry)
                              legacy-content-algorithm)
                          (str (:store-key part)))
                :storage {:format :boring-attention-chunk
                          :byte-order (get-in record [:layout :byte-order])
                          :payload-path [:chunk/payload]}}))
            selected)})))

(defn boundary-manifest
  "Return the certified manifest for the state at the end of `entries`.

  `record` is a catalog layout record for `model-fingerprint`; `entries` is
  the root-first chain ending at the boundary. Throws when Raster's certifier
  rejects the result."
  [record model-fingerprint entries]
  (let [entries (vec entries)
        durable-layout (:layout record)
        attention-layout (:attention-state durable-layout)
        dtype (or (get raster-dtypes (:dtype durable-layout))
                  (throw (ex-info "Unsupported durable storage dtype"
                                  {:dtype (:dtype durable-layout)})))
        _ (when (empty? entries)
            (throw (ex-info "A manifest needs at least one chunk" {})))
        selection (parts/boundary-selection attention-layout entries)
        step (grid-step entries)
        tail (peek entries)]
    (numerical-state/certify
     (numerical-state/manifest
      {:id (catalog/chunk-entry-id model-fingerprint (:kv/prefix-hash tail))
       :parents (if-let [parent (:kv/parent-hash tail)]
                  [(catalog/chunk-entry-id model-fingerprint parent)]
                  [])
       :logical-coordinate {:processed-count (:boundary selection)}
       :fields (mapv #(group-field attention-layout record dtype selection step %)
                     (attention-state/groups attention-layout))
       :numerical-contract {:mode (:kv.layout/numerical-mode record)
                            :determinism (:kv.layout/determinism record)
                            :compatibility-id model-fingerprint}
       :provenance {:program-fingerprint model-fingerprint}}))))

(defn continuation-manifest
  "Return the certified manifest for the continuation state at `prefix-hash`.

  The fingerprint's layout record supplies field shapes, storage dtype, and
  numerical contract; each node names its content-address algorithm. Throws when the chain or
  layout record is missing, or when Raster's certifier rejects the result."
  [database model-fingerprint prefix-hash]
  (let [record (or (catalog/lookup-layout database model-fingerprint)
                   (throw (ex-info "No durable layout is recorded for this fingerprint"
                                   {:model-fingerprint model-fingerprint})))]
    (boundary-manifest record model-fingerprint
                       (chain database model-fingerprint prefix-hash))))

(defn load-plan
  "Return what restoring a certified manifest's state loads.

  `{:parts [{:group :start :token-count :store-key :bytes}] :window-floors}`,
  parts ordered by token start then field order."
  [certified]
  (let [fields (get-in certified [:manifest :fields])]
    {:parts
     (->> fields
          (mapcat
           (fn [field]
             (let [origin (long (get-in field [:coordinate-space :token-origin] 0))]
               (map (fn [chunk]
                      {:group (:id field)
                       :start (+ origin (long (second (:offsets chunk))))
                       :token-count (long (second (:shape chunk)))
                       :store-key (UUID/fromString (get-in chunk [:content :digest]))
                       :bytes (:stored-byte-length chunk)})
                    (:chunks field)))))
          (sort-by :start)
          vec)
     :window-floors
     (into {}
           (for [field fields
                 :when (= :window (get-in field [:value :attributes :extent :kind]))]
             [(:id field) (get-in field [:coordinate-space :token-origin])]))}))
