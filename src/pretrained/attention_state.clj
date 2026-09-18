(ns pretrained.attention-state
  "Declarative persistent attention-state layouts.

  Conventional MHA/GQA/MQA models use K and V slabs. Architectures with another
  recurrent attention representation can provide `:attention-state` in their
  descriptor without changing continuation identity, storage, or cache policy."
  (:import [java.nio ByteOrder]))

(def default-descriptor
  {:kind :kv
   :token-axis :position
   :dtype :float32
   :slabs [{:name :key
            :tensor-key :continuation/keys
            :buffer-prefix "kc"
            :count-key :n-layers
            :feature-keys [:n-kv :head-dim]}
           {:name :value
            :tensor-key :continuation/values
            :buffer-prefix "vc"
            :count-key :n-layers
            :feature-keys [:n-kv :head-dim]}]})

(defn- resolve-number
  [model slab value-key model-key]
  (let [value (or (get slab value-key) (get model (get slab model-key)))]
    (when-not (and (number? value) (pos? (long value)))
      (throw (ex-info "Attention-state slab dimension must be positive"
                      {:slab (:name slab) :dimension value-key :value value})))
    (long value)))

(defn- resolve-elements-per-token
  [model slab]
  (if (some? (:elements-per-token slab))
    (resolve-number model slab :elements-per-token :unused)
    (let [feature-keys (:feature-keys slab)
          values (mapv #(get model %) feature-keys)]
      (when-not (and (seq feature-keys)
                     (every? #(and (number? %) (pos? (long %))) values))
        (throw (ex-info "Attention-state slab features must be positive"
                        {:slab (:name slab) :feature-keys feature-keys
                         :values values})))
      (reduce * 1 (map long values)))))

(defn global-layer?
  "Return true when `layer` attends to the whole prefix.

  From descriptor flags: an explicit `:global-layers` set, else a
  `:global-layer-pattern` p (every p-th layer, Gemma style), else every layer.
  Decoders, rope selection, and attention-state groups all use this one
  predicate, so group membership and attention visibility cannot diverge."
  [model layer]
  (let [flags (get-in model [:desc :flags])]
    (cond
      (:global-layers flags) (contains? (:global-layers flags) layer)
      (:global-layer-pattern flags)
      (zero? (mod (inc (long layer)) (long (:global-layer-pattern flags))))
      :else true)))

(defn layer-window
  "Return the sliding attention window of `layer` in tokens, or nil.

  A window of N tokens means position q attends to positions q-(N-1) through
  q. Global layers, and models without a sliding window, return nil."
  [model layer]
  (when-let [window (get-in model [:desc :flags :sliding-window :size])]
    (when-not (global-layer? model layer)
      (when-not (and (integer? window) (pos? window))
        (throw (ex-info "Sliding attention window must contain at least one token"
                        {:layer layer :window window})))
      (long window))))

(defn min-window
  "Return the smallest sliding window over the model's layers, or nil."
  [model]
  (when-let [windows (seq (keep #(layer-window model %)
                                (range (long (or (:n-layers model) 0)))))]
    (reduce min windows)))

(def ^:private slab-keys
  #{:name :tensor-key :buffer-prefix :count :count-key :elements-per-token
    :feature-keys})

(defn- require-per-token-slab!
  "Reject slab keys the resolver does not implement.

  Every slab is currently sized per token. A descriptor that declares any other
  extent, such as a fixed-size recurrent state, would otherwise be allocated
  once per token."
  [slab]
  (let [unexpected (remove slab-keys (keys slab))]
    (when (seq unexpected)
      (throw (ex-info (str "Attention-state slab declares unsupported keys; only "
                           "per-token slabs are implemented")
                      {:slab (:name slab)
                       :unsupported (vec (sort unexpected))
                       :supported slab-keys})))))

(defn- derive-groups
  "Return retention groups for a model with sliding-window layers, or nil.

  Members are `[slab layer]` pairs in payload order. A member belongs to the
  window group of its layer's window, or to the `:global` token group. Only
  slabs indexed by model layer can be grouped this way."
  [model descriptor-slabs slabs]
  (when (min-window model)
    (doseq [[descriptor slab] (map vector descriptor-slabs slabs)]
      (when-not (and (= :n-layers (:count-key descriptor))
                     (= (long (:count slab)) (long (:n-layers model))))
        (throw (ex-info "Sliding-window layouts require slabs indexed by model layer"
                        {:slab (:name slab) :count-key (:count-key descriptor)}))))
    (let [members (for [slab slabs
                        layer (range (:count slab))]
                    {:member [(:name slab) layer]
                     :window (layer-window model layer)
                     :width (:elements-per-token slab)})
          by-window (group-by :window members)]
      (mapv (fn [window]
              (let [group-members (get by-window window)
                    widths (into #{} (map :width) group-members)]
                (when-not (= 1 (count widths))
                  (throw (ex-info "Attention-state group members need one row width"
                                  {:window window :widths widths})))
                ;; A window of one attends only the current token, so no
                ;; stored row is ever read and no state could be restored.
                (when (and window (< (long window) 2))
                  (throw (ex-info "Sliding windows must span at least two tokens"
                                  {:window window})))
                {:id (if window (keyword (str "window-" window)) :global)
                 :extent (if window {:kind :window :size window} {:kind :token})
                 :members (mapv :member group-members)}))
            (sort-by #(or % -1) (keys by-window))))))

(defn- require-unique!
  [slabs field]
  (let [values (mapv field slabs)]
    (when-not (= (count values) (count (set values)))
      (throw (ex-info "Attention-state slab identifiers must be unique"
                      {:field field :values values})))))

(defn layout
  "Return a validated, fully resolved attention-state layout for `model`.

  A descriptor slab supplies `:name`, `:tensor-key`, `:buffer-prefix`, and either
  a concrete `:count`/`:elements-per-token` or model-key recipes via `:count-key`
  and `:feature-keys`. Returned slabs contain only resolved numeric dimensions."
  [model]
  (let [descriptor (merge default-descriptor (get-in model [:desc :attention-state]))
        _ (when-not (= :position (:token-axis descriptor))
            (throw (ex-info "Attention-state layouts must use the :position token axis"
                            {:token-axis (:token-axis descriptor)})))
        _ (run! require-per-token-slab! (:slabs descriptor))
        slabs
        (mapv (fn [slab]
                (let [count (resolve-number model slab :count :count-key)
                      elements-per-token (resolve-elements-per-token model slab)]
                  (when-not (and (:name slab) (:tensor-key slab) (:buffer-prefix slab)
                                 (number? elements-per-token)
                                 (pos? (long elements-per-token)))
                    (throw (ex-info "Attention-state slab descriptor is incomplete"
                                    {:slab slab})))
                  (-> slab
                      (dissoc :count-key :feature-keys)
                      (assoc :count count
                             :elements-per-token (long elements-per-token)))))
              (:slabs descriptor))
        groups (derive-groups model (:slabs descriptor) slabs)]
    (when-not (seq slabs)
      (throw (ex-info "Attention-state layout requires at least one slab" {})))
    (doseq [field [:name :tensor-key :buffer-prefix]]
      (require-unique! slabs field))
    ;; `:groups` is present only when retention differs across members, so a
    ;; model without windows keeps the exact layout, and therefore the
    ;; fingerprint and chunk content ids, it had before groups existed.
    (cond-> {:version 1
             :kind (:kind descriptor)
             :token-axis (:token-axis descriptor)
             :dtype (:dtype descriptor)
             :byte-order (if (= ByteOrder/LITTLE_ENDIAN (ByteOrder/nativeOrder))
                           :little-endian :big-endian)
             :slabs slabs}
      groups (assoc :groups groups))))

(defn first-attended-row
  "Return the first row the token at `boundary` attends to in `group`.

  A token at position b attends to rows [b - (w - 1), b] of a window-w group
  and to every earlier row of a token group. This is the one statement of
  that bound: part selection, window floors and fork checks all use it."
  [group boundary]
  (case (get-in group [:extent :kind])
    :token 0
    :window (max 0 (- (long boundary) (dec (long (get-in group [:extent :size])))))
    (throw (ex-info "Unsupported attention-state group extent"
                    {:group (:id group) :extent (:extent group)}))))

(def implicit-group-id
  "The id of the single token group of a layout without `:groups`."
  :attention-state)

(defn groups
  "Return the retention groups of a resolved `layout`.

  A layout without `:groups` has one implicit token group holding every
  `[slab layer]` member in payload order."
  [layout]
  (or (:groups layout)
      [{:id implicit-group-id
        :extent {:kind :token}
        :members (vec (for [slab (:slabs layout)
                            layer (range (:count slab))]
                        [(:name slab) layer]))}]))

(defn group
  "Return the group of `layout` named `id`, or throw."
  [layout id]
  (or (some #(when (= id (:id %)) %) (groups layout))
      (throw (ex-info "Attention-state layout has no such group"
                      {:group id :groups (mapv :id (groups layout))}))))

(defn group-payload-plan
  "Describe the contiguous payload of one group for `token-count` rows.

  Entries follow the group's member order with the same shape as
  `payload-plan`. For the implicit group the result equals `payload-plan`."
  [layout group token-count]
  (let [by-name (into {} (map (juxt :name identity)) (:slabs layout))]
    (loop [members (:members group)
           offset 0
           result []]
      (if-let [[slab-name layer] (first members)]
        (let [elements (* (long token-count)
                          (long (:elements-per-token (get by-name slab-name))))]
          (recur (next members) (+ offset elements)
                 (conj result {:slab slab-name
                               :layer layer
                               :element-offset offset
                               :elements elements})))
        result))))

(defn buffer-key
  "Return the resident GPU buffer key for resolved `slab` and `layer`."
  [slab layer]
  (keyword (str (:buffer-prefix slab) (long layer))))

(defn tensor-groups
  "Return resolved slabs paired with their runtime tensor vectors from `state`."
  [state]
  (mapv (fn [slab]
          (let [tensors (get state (:tensor-key slab))]
            (when-not (= (:count slab) (count tensors))
              (throw (ex-info "Continuation tensors do not match attention-state layout"
                              {:slab (:name slab) :expected (:count slab)
                               :actual (count tensors)})))
            [slab tensors]))
        (get-in state [:continuation/layout :attention-state :slabs])))

(defn payload-plan
  "Describe the contiguous payload slices for `token-count` rows.

  Order is descriptor slab order, then layer order. Returned entries contain
  `:slab`, `:layer`, `:element-offset`, and `:elements`."
  [layout token-count]
  (loop [remaining (for [slab (:slabs layout)
                        layer (range (:count slab))]
                    [slab layer])
         offset 0
         result []]
    (if-let [[slab layer] (first remaining)]
      (let [elements (* (long token-count) (:elements-per-token slab))]
        (recur (next remaining) (+ offset elements)
               (conj result {:slab (:name slab)
                             :layer layer
                             :element-offset offset
                             :elements elements})))
      result)))
