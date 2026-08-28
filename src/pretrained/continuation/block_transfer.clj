(ns pretrained.continuation.block-transfer
  "Dense resident staging for fragmented paged KV routes.

  One composed Raster LinkPlan gathers or scatters every attention-state
  slab/layer. The durable representation stays dense while the page index
  vector selects arbitrary worker-local physical pages."
  (:refer-clojure :exclude [run!])
  (:require [raster.compiler.ir.link-plan :as link]
            [raster.compiler.pipeline :as pipeline]
            [raster.core :as raster :refer [deftm]]
            [raster.dl.array-ops :as array-ops]
            [raster.gpu.core :as gpu]
            [raster.gpu.link :as gpu-link])
  (:import [java.util UUID]))

#_{:clj-kondo/ignore [:unresolved-symbol]}
(deftm scatter-half-blocks!
  [src :- (Array short), indices :- (Array int), out :- (Array short),
   nblocks :- Long, block-width :- Long, indexed-blocks :- Long] :- Void
  (array-ops/scatter-blocks! src indices out nblocks block-width indexed-blocks))

#_{:clj-kondo/ignore [:unresolved-symbol]}
(deftm gather-half-blocks!
  [src :- (Array short), indices :- (Array int), out :- (Array short),
   nblocks :- Long, block-width :- Long, indexed-blocks :- Long] :- Void
  (array-ops/gather-blocks! src indices out nblocks block-width indexed-blocks))

(defrecord BlockTransferEngine
           [session nblocks index-key staging-keys executables])

(defn engine?
  "Return true when `value` is a fragmented block-transfer engine."
  [value]
  (instance? BlockTransferEngine value))

(defn- compiled-descriptor*
  [direction device-id]
  (let [program (case direction
                  :scatter #'scatter-half-blocks!
                  :gather #'gather-half-blocks!)]
    (pipeline/compile-gpu-program
     (raster/resolve-deftm-var program {:dtype :float})
     device-id :dtype :float)))

(def ^:private compiled-descriptor
  (memoize compiled-descriptor*))

(defn- node
  [id dtype elements device-id role allocation-id]
  (link/node {:id id
              :dtype dtype
              :shape [(long elements)]
              :device device-id
              :role role
              :ownership :external
              :allocation-id allocation-id}))

(defn- transfer-plan
  [id device-id descriptor layout page-size physical-pages buffer-keys
   staging-keys index-key nblocks direction]
  (let [index-node :block-indices
        layer-specs
        (vec
         (for [slab (:slabs layout)
               layer (range (:count slab))
               :let [slab-layer [(:name slab) layer]
                     block-width (* (long page-size)
                                    (long (:elements-per-token slab)))]]
           {:slab-layer slab-layer
            :block-width block-width
            :staging-node [:staging slab-layer]
            :pool-node [:pool slab-layer]}))
        nodes
        (into
         [(node index-node :int nblocks device-id :state index-key)]
         (mapcat
          (fn [{:keys [slab-layer block-width staging-node pool-node]}]
            [(node staging-node :half (* nblocks block-width) device-id
                   :state (get staging-keys slab-layer))
             (node pool-node :half (* physical-pages block-width) device-id
                   :state (get buffer-keys slab-layer))])
          layer-specs))
        instances
        (mapv
         (fn [{:keys [slab-layer block-width staging-node pool-node]}]
           (link/instance
            {:id [direction slab-layer]
             :descriptor descriptor
             :bindings
             (case direction
               :scatter {'src staging-node 'indices index-node 'out pool-node}
               :gather {'src pool-node 'indices index-node 'out staging-node})
             :scalars {'nblocks nblocks
                       'block-width block-width
                       'indexed-blocks physical-pages}}))
         layer-specs)
        outputs (mapv (case direction
                        :scatter :pool-node
                        :gather :staging-node)
                      layer-specs)]
    (link/make {:id id
                :target device-id
                :nodes nodes
                :instances instances
                :outputs outputs
                :attributes {:owner :pretrained.continuation.block-transfer
                             :direction direction}})))

(defn open!
  "Allocate dense FP16 staging and compose Raster gather/scatter graphs.

  The engine borrows the page-pool buffers, shares one staging workspace across
  both directions, and owns its linked executables. Call `close!` before closing
  the owning Raster session."
  [session layout page-size physical-pages buffer-keys nblocks]
  (when-not (pos-int? nblocks)
    (throw (ex-info "Block-transfer extent is invalid"
                    {:nblocks nblocks})))
  (let [device-id (:device-id @session)
        prefix (str "kv-block-" (UUID/randomUUID))
        index-key (keyword (str prefix "-indices"))
        staging-keys
        (into {}
              (for [slab (:slabs layout)
                    layer (range (:count slab))]
                [[(:name slab) layer]
                 (keyword (str prefix "-" (name (:name slab)) "-" layer))]))
        allocations
        (into
         {index-key [:int nblocks nil :state]}
         (for [slab (:slabs layout)
               layer (range (:count slab))
               :let [elements (* (long nblocks) (long page-size)
                                 (long (:elements-per-token slab)))]]
           [(get staging-keys [(:name slab) layer])
            [:half elements nil :state]]))
        opened (atom [])]
    (try
      (gpu/alloc! session allocations)
      (let [allocation-keys (concat [index-key] (vals staging-keys))
            external-buffers
            (into {}
                  (map (fn [key] [key (gpu/buffer session key)]))
                  (concat allocation-keys (vals buffer-keys)))
            executables
            (into {}
                  (for [direction [:scatter :gather]
                        :let [plan (transfer-plan
                                    [:paged-kv-block direction nblocks
                                     (random-uuid)]
                                    device-id
                                    (compiled-descriptor direction device-id)
                                    layout page-size physical-pages buffer-keys
                                    staging-keys index-key nblocks direction)
                              executable
                              (gpu-link/instantiate!
                               plan {:session session
                                     :external-buffers external-buffers})
                              _ (swap! opened conj executable)]]
                    [direction executable]))]
        (->BlockTransferEngine session nblocks index-key staging-keys executables))
      (catch Throwable error
        (doseq [executable (reverse @opened)]
          (try (gpu-link/close! executable) (catch Throwable _)))
        (doseq [key (keys allocations)]
          (when (gpu/buffer session key)
            (gpu/free-buffer! session key)))
        (throw error)))))

(defn index-buffer-key
  "Return the engine's resident physical-page index buffer key."
  [engine]
  (:index-key engine))

(defn staging-buffer-key
  "Return the dense staging buffer key for `[slab-name layer]`."
  [engine slab-name layer]
  (or (get (:staging-keys engine) [slab-name layer])
      (throw (ex-info "Block-transfer staging slab/layer is absent"
                      {:slab slab-name :layer layer}))))

(defn run!
  "Run the engine's composed resident `:gather` or `:scatter` graph once."
  [engine direction]
  (if-let [executable (get (:executables engine) direction)]
    (gpu-link/run! executable)
    (throw (ex-info "Block-transfer direction is invalid"
                    {:direction direction}))))

(defn close!
  "Release the linked graph and owned staging buffers. Idempotence is delegated
  to Raster's linked executable and session buffer lifecycle."
  [engine]
  (doseq [executable (reverse (vals (:executables engine)))]
    (gpu-link/close! executable))
  (doseq [key (concat [(:index-key engine)] (vals (:staging-keys engine)))]
    (when (gpu/buffer (:session engine) key)
      (gpu/free-buffer! (:session engine) key)))
  nil)
