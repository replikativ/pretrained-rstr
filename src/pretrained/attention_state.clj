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
              (:slabs descriptor))]
    (when-not (seq slabs)
      (throw (ex-info "Attention-state layout requires at least one slab" {})))
    (doseq [field [:name :tensor-key :buffer-prefix]]
      (require-unique! slabs field))
    {:version 1
     :kind (:kind descriptor)
     :token-axis (:token-axis descriptor)
     :dtype (:dtype descriptor)
     :byte-order (if (= ByteOrder/LITTLE_ENDIAN (ByteOrder/nativeOrder))
                   :little-endian :big-endian)
     :slabs slabs}))

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
