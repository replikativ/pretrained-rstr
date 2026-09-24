(ns pretrained.continuation.parts
  "Which stored parts a continuation boundary needs.

  A catalog node holds one chain position. A layout with one retention group
  stores one blob per node; a layout with several groups stores one part per
  group. Restoring the state at boundary b needs every token-group part, but a
  window group of size w needs only the parts that intersect the rows the next
  token attends to, [b - w + 1, b). Restore, the certified manifest, and the
  router's byte estimate all select through this namespace, so they cannot
  disagree about what a boundary contains."
  (:require [pretrained.attention-state :as attention-state]))

(defn entry-parts
  "Return the stored parts of a catalog chunk entry as
  `[{:group :store-key :bytes}]`, in the entry's canonical group order.

  A single-group node has one implicit part carried on the node itself."
  [entry]
  (if-let [parts (seq (:kv/parts entry))]
    (->> parts
         (map (fn [part]
                {:group (:kv.part/group part)
                 :store-key (:kv.part/store-key part)
                 :bytes (:kv.part/bytes part)}))
         (sort-by (comp str :group))
         vec)
    (do
      (when-not (:kv/store-key entry)
        (throw (ex-info "Catalog chunk entry has neither a blob nor parts"
                        {:prefix-hash (:kv/prefix-hash entry)})))
      [{:group attention-state/implicit-group-id
        :store-key (:kv/store-key entry)
        :bytes (:kv/bytes entry)}])))

(defn boundary-selection
  "Select the parts of root-first `entries` needed at the chain's end.

  Returns `{:boundary :chunks :window-floors}`. Each chunk is
  `{:entry :start :token-count :parts}` with only the needed parts;
  `:window-floors` maps each window group to the first row it holds. Throws
  when an entry's parts do not match the layout's groups."
  [layout entries]
  (let [groups (attention-state/groups layout)
        group-ids (into #{} (map :id) groups)
        by-id (into {} (map (juxt :id identity)) groups)
        tail (peek (vec entries))
        boundary (if tail
                   (+ (long (:kv/start-token tail)) (long (:kv/token-count tail)))
                   0)
        lowest (into {} (map (fn [g] [(:id g) (attention-state/first-attended-row g boundary)]))
                     groups)
        chunks
        (mapv (fn [entry]
                (let [start (long (:kv/start-token entry))
                      token-count (long (:kv/token-count entry))
                      end (+ start token-count)
                      parts (entry-parts entry)]
                  (when-not (= group-ids (into #{} (map :group) parts))
                    (throw (ex-info "Catalog chunk parts do not match the layout groups"
                                    {:prefix-hash (:kv/prefix-hash entry)
                                     :parts (mapv :group parts)
                                     :groups group-ids})))
                  {:entry entry
                   :start start
                   :token-count token-count
                   :parts (filterv #(> end (long (get lowest (:group %)))) parts)}))
              entries)
        window-floors
        (into {}
              (for [group groups
                    :when (= :window (get-in group [:extent :kind]))]
                [(:id group)
                 (or (some (fn [{:keys [start parts]}]
                             (when (some #(= (:id group) (:group %)) parts) start))
                           chunks)
                     boundary)]))]
    {:boundary boundary
     :chunks chunks
     :window-floors window-floors
     :groups (mapv #(select-keys (get by-id %) [:id :extent :members])
                   (map :id groups))}))

(defn selected-bytes
  "Return the stored bytes a selection loads."
  [selection]
  (reduce + 0 (for [chunk (:chunks selection)
                    part (:parts chunk)]
                (long (:bytes part)))))
