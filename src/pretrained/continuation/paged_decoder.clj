(ns pretrained.continuation.paged-decoder
  "Paged autoregressive execution over an already-bound resident decoder.

  Raster links the generated layer stages, paged K/V append, routed attention,
  and token head into one resident executable. Q, K, V, and attention output
  stay in device buffers; the host moves only routing metadata and the selected
  token."
  (:refer-clojure :exclude [run!])
  (:require [pretrained.attention-state :as attention-state]
            [pretrained.continuation.page-pool :as page-pool]
            [pretrained.continuation.paged-append :as paged-append]
            [pretrained.continuation.paged-attention :as paged-attention]
            [pretrained.decoder-gpu :as decoder-gpu]
            [raster.compiler.ir.attention :as attention]
            [raster.compiler.ir.kernel-executable :as kernel-executable]
            [raster.compiler.ir.link-plan :as link]
            [raster.gpu.core :as gpu]
            [raster.gpu.link :as gpu-link])
  (:import [java.io Closeable]
           [java.util UUID]))

(declare close!)

(defrecord PagedDecoder
           [decode-state pool executable descriptor-keys pages-per-sequence
            attention-execution prefill-executable prefill-descriptor-keys
            prefill-T state]
  Closeable
  (close [decoder]
    (close! decoder)))

(defn paged-decoder?
  "Return true when `value` is a paged decoder."
  [value]
  (instance? PagedDecoder value))

(defn- checked-positive
  [field value]
  (when-not (and (integer? value) (pos? value))
    (throw (ex-info "Paged decoder capacity must be a positive integer"
                    {:field field :value value})))
  (long value))

(defn- require-open!
  [decoder]
  (when (:closed? @(:state decoder))
    (throw (ex-info "Paged decoder is closed" {}))))

(defn- validate-staged-executables!
  [model execution include-head-tail?]
  (let [layers (:layers execution)]
    (when-not (and (= (:n-layers model) (count layers))
                   (every? #(and (gpu-link/linked-executable? (:pre %))
                                 (gpu-link/linked-executable? (:post %)))
                           layers)
                   (or (not include-head-tail?)
                       (gpu-link/linked-executable? (:head-tail execution))))
      (throw (ex-info "Decode state has no complete staged LinkPlan execution"
                      {:stage-executables execution
                       :include-head-tail? include-head-tail?})))
    execution))

(defn- staged-executables!
  [decode-state]
  (let [execution (:stage-executables decode-state)]
    (validate-staged-executables! (:model decode-state) execution true)))

(defn- graph-instance
  [id graph graph-bindings]
  (let [abi (:abi (kernel-executable/validate! graph))
        arguments (kernel-executable/arguments graph)
        entries
        (mapv (fn [index slot argument]
                (let [parameter (symbol (str "argument-" index))]
                  {:parameter parameter
                   :slot slot
                   :argument argument
                   :node-id (when-not (= :scalar (:kind slot))
                              (let [binding (get graph-bindings argument)]
                                (if (gpu/resident-buffer-view? binding)
                                  (:key binding)
                                  binding)))}))
              (range) abi arguments)
        pointer-entries (filterv #(not= :scalar (get-in % [:slot :kind])) entries)
        scalar-entries (filterv #(= :scalar (get-in % [:slot :kind])) entries)
        descriptor
        {:dtype nil
         :all-params (mapv :parameter entries)
         :array-params (mapv :parameter pointer-entries)
         :scalar-params (mapv :parameter scalar-entries)
         :array-roles
         (into {}
               (map (fn [{:keys [parameter slot]}]
                      [parameter (if (= :input (:kind slot)) :input :output)]))
               pointer-entries)
         :allocs []
         :steps
         [{:phase id
           :kernel-name (str (name (if (keyword? id) id :linked-graph)))
           :convention :map
           :artifact graph
           :argument-specs
           (mapv (fn [index {:keys [parameter slot]}]
                   (if (= :scalar (:kind slot))
                     {:kind :scalar
                      :sym parameter
                      :type (:kernel-dtype slot)
                      :value-fn #(nth % index)}
                     {:kind (:kind slot) :sym parameter}))
                 (range) entries)}]
         :result-sym (some-> pointer-entries last :parameter)}]
    {:instance
     (link/instance
      {:id id
       :descriptor descriptor
       :bindings (into {} (map (juxt :parameter :node-id)) pointer-entries)
       :scalars
       (into {}
             (map (fn [{:keys [parameter argument]}]
                    [parameter
                     (or (get graph-bindings argument)
                         (throw (ex-info "Linked graph scalar has no bound value"
                                         {:instance id :argument argument})))]))
             scalar-entries)})
     :buffers
     (into {}
           (map (juxt :id identity))
           (concat (:inputs graph) (:outputs graph)))
     :node-ids
     (into {} (map (juxt :argument :node-id)) pointer-entries)}))

(defn- graph-nodes
  [device-id graph-component roles]
  (into {}
        (map (fn [[argument node-id]]
               (let [buffer (get-in graph-component [:buffers argument])
                     elements (:elements buffer)]
                 (when-not (integer? elements)
                   (throw (ex-info "Paged graph has an unresolved external extent"
                                   {:argument argument :elements elements})))
                 [node-id
                  (link/node {:id node-id
                              :dtype (:dtype buffer)
                              :shape [(long elements)]
                              :device device-id
                              :role (get roles node-id :internal)
                              :ownership :external
                              :allocation-id node-id})])))
        (:node-ids graph-component)))

(defn- merge-first
  [maps]
  (reduce (fn [result values]
            (reduce-kv #(if (contains? %1 %2) %1 (assoc %1 %2 %3)) result values))
          {} maps))

(defn- global-layer?
  [model layer]
  (let [flags (get-in model [:desc :flags])]
    (cond
      (:global-layers flags) (contains? (:global-layers flags) layer)
      (:global-layer-pattern flags)
      (zero? (mod (inc (long layer)) (long (:global-layer-pattern flags))))
      :else true)))

(defn- layer-visibility
  [model layer]
  (if-let [window (and (not (global-layer? model layer))
                       (get-in model [:desc :flags :sliding-window :size]))]
    (do
      (when-not (and (integer? window) (pos? window))
        (throw (ex-info "Sliding attention window must contain at least one token"
                        {:layer layer :window window})))
      ;; Raster windows are inclusive distances. A model window of N tokens
      ;; therefore admits q-(N-1) through q.
      (attention/visibility {:causal? true :window-left (dec (long window))}))
    (attention/visibility {:causal? true})))

(defn- linked-paged-executable!
  [decode-state pool row-count pages-per-sequence prefix query-view positions-view
   key-view value-view output-view
   {:keys [attention-schedule history-tile-size staged sequence-count
           include-head-tail? input-node input-role output-node plan-id]
    :or {include-head-tail? true input-node :r0 input-role :state
         plan-id ::decoder}}]
  (let [{:keys [sess model device-id device-desc]} decode-state
        sequence-count (long (or sequence-count row-count))
        staged (validate-staged-executables!
                model (or staged (:stage-executables decode-state))
                include-head-tail?)
        output-node (or output-node (when include-head-tail? :tokbuf))
        slot-prefix (str prefix "-append")
        route-prefix (str prefix "-attention")
        append-plans
        (mapv (fn [layer]
                (paged-append/reference-plan
                 pool {:id [::append layer]
                       :key-prefix slot-prefix
                       :layer layer
                       :batch-size row-count
                       :key-view key-view
                       :value-view value-view}))
              (range (:n-layers model)))
        attention-plans
        (mapv (fn [layer]
                (paged-attention/reference-plan
                 pool {:id [::attention layer]
                       :key-prefix route-prefix
                       :layer layer
                       :batch-size sequence-count
                       :total-query-tokens row-count
                       :q-heads (:n-q model)
                       :kv-heads (:n-kv model)
                       :qk-head-dim (:head-dim model)
                       :value-head-dim (:head-dim model)
                       :pages-per-sequence pages-per-sequence
                       :scale (:attn-scale model)
                       :visibility (layer-visibility model layer)
                       :device-desc device-desc
                       :attention-schedule attention-schedule
                       :history-tile-size history-tile-size
                       :query-dtype :float
                       :output-dtype :float
                       :query-view query-view
                       :query-positions-view positions-view
                       :output-view output-view}))
              (range (:n-layers model)))
        descriptor-specs
        (merge (into {} (map :allocation) append-plans)
               (apply merge (map :allocations attention-plans)))
        descriptor-keys (set (keys descriptor-specs))
        page-keys (set (vals (page-pool/buffer-keys pool)))]
    (gpu/alloc! sess (into {} (map (fn [[key spec]] [key (vec (take 3 spec))]))
                            descriptor-specs))
    (try
      (let [append-components
            (mapv (fn [layer plan]
                    (graph-instance [::append layer] (:graph plan) (:bindings plan)))
                  (range) append-plans)
            attention-components
            (mapv (fn [layer plan]
                    (graph-instance [::attention layer] (:graph plan) (:bindings plan)))
                  (range) attention-plans)
            stage-plans
            (cond->
             (vec (mapcat (fn [{:keys [pre post]}] [(:plan pre) (:plan post)])
                          (:layers staged)))
              include-head-tail? (conj (:plan (:head-tail staged))))
            stage-nodes (merge-first (map :nodes stage-plans))
            graph-roles (merge (zipmap descriptor-keys (repeat :input))
                               {(:key positions-view) :input
                                input-node input-role}
                               (zipmap page-keys (repeat :state))
                               (zipmap [(:key query-view) (:key key-view)
                                        (:key value-view) (:key output-view)]
                                       (repeat :internal)))
            append-nodes (map #(graph-nodes device-id % graph-roles) append-components)
            attention-nodes (map #(graph-nodes device-id % graph-roles) attention-components)
            nodes (reduce-kv (fn [result node-id role]
                               (if-let [node (get result node-id)]
                                 (assoc result node-id (assoc node :role role))
                                 result))
                             (merge-first
                              (concat [stage-nodes] append-nodes attention-nodes))
                             graph-roles)
            instances
            (vec
             (concat
              (mapcat
               (fn [layer]
                 (concat
                  (:instances (:plan (get-in staged [:layers layer :pre])))
                  [(get-in append-components [layer :instance])
                   (get-in attention-components [layer :instance])]
                  (:instances (:plan (get-in staged [:layers layer :post])))))
               (range (:n-layers model)))
              (when include-head-tail?
                (:instances (:plan (:head-tail staged))))))
            aliases (into #{} (mapcat :aliases) stage-plans)
            plan (link/make
                  {:id [plan-id prefix]
                   :target device-id
                   :nodes nodes
                   :instances instances
                   :outputs [output-node]
                   :aliases aliases
                   :attributes {:owner ::paged-decoder}})
            allocation-ids
            (into #{} (map #(get-in % [:view :allocation :id])) (vals (:nodes plan)))
            external-buffers
            (into {}
                  (map (fn [allocation-id]
                         [allocation-id
                          (or (gpu/buffer sess allocation-id)
                              (throw (ex-info "Paged LinkPlan allocation is not resident"
                                              {:allocation allocation-id})))]))
                  allocation-ids)
            layer-execution (mapv paged-attention/execution-summary
                                  attention-plans)]
        {:executable (gpu-link/instantiate!
                      plan {:session sess :external-buffers external-buffers})
         :descriptor-keys
         {:slots (:slot-key (first append-plans))
          :row-offsets (get-in (first attention-plans) [:buffer-keys :query-row-offsets])
          :positions (get-in (first attention-plans) [:buffer-keys :query-positions])
          :page-table (get-in (first attention-plans) [:buffer-keys :page-table])
          :lengths (get-in (first attention-plans) [:buffer-keys :lengths])
          :start-positions (get-in (first attention-plans)
                                   [:buffer-keys :start-positions])}
         :attention-execution
         {:layers layer-execution
          :strategies (frequencies (map :strategy layer-execution))
          :temporary-bytes (reduce + 0 (map :temporary-bytes layer-execution))}
         :owned-buffer-keys descriptor-keys})
      (catch Throwable error
        (doseq [key descriptor-keys]
          (when (gpu/buffer sess key)
            (gpu/free-buffer! sess key)))
        (throw error)))))

(defn open!
  "Attach paged K/V execution to a resident state returned by `bind-decode!`.

  Options:
  - `:page-size` defaults to 16 tokens.
  - `:physical-pages` defaults to enough pages for one full route per batch lane.
  - `:key-prefix` optionally supplies deterministic session buffer names.
  - `:attention-schedule` selects Raster's `:auto`, `:reference`,
    `:subgroup-online-score-reuse`, or `:subgroup-online-tiled-history` policy.
  - `:history-tile-size` overrides the tiled schedule's default and is rejected
    for other policies.

  When `decode-state` was bound with `:prefill-T`, opening also links a
  single-sequence, multi-row causal prefill graph over the same page pool.

  The decoder owns its linked graph but not the decode state, Raster session, or
  page-pool allocations. Closing it releases the graph and descriptor buffers;
  close the decode state's session to release all resident tensors."
  [decode-state & {:keys [page-size physical-pages key-prefix attention-schedule
                          history-tile-size]
                   :or {page-size 16}}]
  (when-not (= :paged (:cache-mode decode-state))
    (throw (ex-info "Paged decoder requires bind-decode! with :cache-mode :paged"
                    {:cache-mode (:cache-mode decode-state)})))
  (let [{:keys [sess model maxpos]} decode-state
        _ (staged-executables! decode-state)
        page-size (checked-positive :page-size page-size)
        pages-per-sequence (long (quot (+ (long maxpos) (dec page-size)) page-size))
        batch-size (long (:batch-size decode-state 1))
        physical-pages (checked-positive
                        :physical-pages
                        (or physical-pages (* batch-size pages-per-sequence)))
        prefix (or key-prefix (str "paged-decoder-" (UUID/randomUUID)))
        pool (page-pool/open-pool!
              sess (attention-state/layout model)
              {:page-size page-size
               :physical-pages physical-pages
               :dtype :half
               :key-prefix (str prefix "-pool")})
        q-elements (* batch-size (long (:n-q model)) (long (:head-dim model)))
        kv-elements (* batch-size (long (:n-kv model)) (long (:head-dim model)))
        query-view (gpu/buffer-view sess :qr {:shape [q-elements]
                                              :id [prefix :query]})
        positions-view (gpu/buffer-view sess :positions {:shape [batch-size]
                                                         :id [prefix :positions]})
        key-view (gpu/buffer-view sess :kr {:shape [kv-elements]
                                            :id [prefix :key]})
        value-view (gpu/buffer-view sess :v {:shape [kv-elements]
                                             :id [prefix :value]})
        output-view (gpu/buffer-view sess :at {:shape [q-elements]
                                               :id [prefix :output]})
        decode-link
        (linked-paged-executable!
         decode-state pool batch-size pages-per-sequence prefix
         query-view positions-view key-view value-view output-view
         {:attention-schedule attention-schedule
          :history-tile-size history-tile-size})
        prefill-T (some-> (:prefill-T decode-state) long)
        prefill-link
        (when prefill-T
          (let [prefill-prefix (str prefix "-prefill")
                prefill-q-elements (* prefill-T (long (:n-q model))
                                      (long (:head-dim model)))
                prefill-kv-elements (* prefill-T (long (:n-kv model))
                                       (long (:head-dim model)))]
            (linked-paged-executable!
             decode-state pool prefill-T pages-per-sequence prefill-prefix
             (gpu/buffer-view sess :pf-qr {:shape [prefill-q-elements]
                                            :id [prefill-prefix :query]})
             (gpu/buffer-view sess :pf-positions {:shape [prefill-T]
                                                   :id [prefill-prefix :positions]})
             (gpu/buffer-view sess :pf-kr {:shape [prefill-kv-elements]
                                            :id [prefill-prefix :key]})
             (gpu/buffer-view sess :pf-v {:shape [prefill-kv-elements]
                                           :id [prefill-prefix :value]})
             (gpu/buffer-view sess :pf-at {:shape [prefill-q-elements]
                                            :id [prefill-prefix :output]})
             {:attention-schedule attention-schedule
              :history-tile-size history-tile-size
              :staged (:prefill-stage-executables decode-state)
              :sequence-count 1
              :include-head-tail? false
              :input-node :pr0
              :input-role :input
              :output-node (keyword (str "pr" (:n-layers model)))
              :plan-id ::prefill})))]
    (map->PagedDecoder
     {:decode-state decode-state
      :pool pool
      :executable (:executable decode-link)
      :descriptor-keys (:descriptor-keys decode-link)
      :pages-per-sequence pages-per-sequence
      :attention-execution (:attention-execution decode-link)
      :prefill-executable (:executable prefill-link)
      :prefill-descriptor-keys (:descriptor-keys prefill-link)
      :prefill-T prefill-T
      :state (atom {:closed? false
                    :owned-buffer-keys
                    (into (:owned-buffer-keys decode-link)
                          (:owned-buffer-keys prefill-link))})})))

(defn attention-execution
  "Return selected per-layer attention strategies and temporary workspace.

  The immutable result is derived from Raster's verified plans when `decoder`
  is opened and contains no runtime handles. Throws when the decoder is closed."
  [decoder]
  (require-open! decoder)
  (:attention-execution decoder))

(defn allocate-continuation!
  "Allocate an empty resident route for `continuation-id` and return it.

  `:start-position` defaults to zero and permits restored context windows whose
  first cached token has a nonzero absolute position."
  [decoder continuation-id & {:keys [start-position]
                              :or {start-position 0}}]
  (require-open! decoder)
  (page-pool/allocate-route! (:pool decoder) continuation-id 0
                             {:start-position start-position}))

(defn prime-token!
  "Put `token` in the shared resident input row without advancing its route."
  [decoder token]
  (require-open! decoder)
  (when-not (= 1 (long (:batch-size (:decode-state decoder) 1)))
    (throw (ex-info "Use prime-tokens! with a batched paged decoder"
                    {:batch-size (:batch-size (:decode-state decoder))})))
  (decoder-gpu/prime-resident-token! (:decode-state decoder) token)
  decoder)

(defn prime-tokens!
  "Put one pending token in every fixed decode lane without advancing routes.

  `tokens` must have the batch size used by `decoder-gpu/bind-decode!`. Returns
  `decoder`."
  [decoder tokens]
  (require-open! decoder)
  (decoder-gpu/prime-resident-tokens! (:decode-state decoder) tokens)
  decoder)

(defn prime-lanes!
  "Upload pending token embeddings only for newly occupied decode lanes.

  Entries are `{:lane index :token token-id}`. Rows omitted from `lane-tokens`
  remain untouched, so continuations retained across iterations keep the next
  embedding produced by the resident decoder tail. Returns `decoder`."
  [decoder lane-tokens]
  (require-open! decoder)
  (decoder-gpu/prime-resident-lanes! (:decode-state decoder) lane-tokens)
  decoder)

(defn- checked-active-lanes
  [decoder lane-work]
  (let [batch-size (long (:batch-size (:decode-state decoder) 1))
        lane-work (->> lane-work
                       (mapv #(update % :position long))
                       (sort-by :lane)
                       vec)
        lanes (mapv :lane lane-work)
        continuation-ids (mapv :continuation-id lane-work)]
    (when-not (seq lane-work)
      (throw (ex-info "Paged execution requires at least one active lane" {})))
    (when-not (and (every? #(and (integer? %) (<= 0 (long %) (dec batch-size))) lanes)
                   (= (count lanes) (count (set lanes))))
      (throw (ex-info "Active decode lanes are invalid for the bound decoder"
                      {:batch-size batch-size :lanes lanes})))
    (when (or (some nil? continuation-ids)
              (not= (count continuation-ids) (count (set continuation-ids))))
      (throw (ex-info "Active decode continuation identities must be unique"
                      {:continuation-ids continuation-ids})))
    lane-work))

(defn- scatter-route-values
  [batch-size pages-per-sequence lanes route-values]
  (let [page-table (int-array (* batch-size pages-per-sequence) -1)
        lengths (int-array batch-size)
        start-positions (int-array batch-size)
        compact-pages ^ints (:page-table route-values)
        compact-lengths ^ints (:lengths route-values)
        compact-starts ^ints (:start-positions route-values)]
    (doseq [[source-row lane] (map-indexed vector lanes)]
      (System/arraycopy compact-pages (* (long source-row) pages-per-sequence)
                        page-table (* (long lane) pages-per-sequence)
                        pages-per-sequence)
      (aset lengths (int lane) (aget compact-lengths source-row))
      (aset start-positions (int lane) (aget compact-starts source-row)))
    {:page-table page-table
     :lengths lengths
     :start-positions start-positions}))

(defn step-lanes!
  "Advance any nonempty subset of a fixed paged decode batch.

  Each work item is `{:lane index :continuation-id id :position absolute-pos}`.
  Active routes reserve and publish one K/V row; inactive lanes receive append
  slot `-1` and an empty attention route, so they consume no pages and mutate no
  continuation. Results are lane-ordered maps with an added `:token`."
  [decoder lane-work]
  (require-open! decoder)
  (let [{:keys [sess batch-size]} (:decode-state decoder)
        batch-size (long (or batch-size 1))
        lane-work (checked-active-lanes decoder lane-work)
        lanes (mapv :lane lane-work)
        continuation-ids (mapv :continuation-id lane-work)]
    (doseq [{:keys [continuation-id position]} lane-work]
      (let [route (or (page-pool/route (:pool decoder) continuation-id)
                      (throw (ex-info "Continuation is not resident"
                                      {:continuation-id continuation-id})))
            expected-position (+ (long (:start-position route))
                                 (long (:token-count route)))]
        (when-not (= expected-position position)
          (throw (ex-info "Paged decode position does not extend the resident route"
                          {:continuation-id continuation-id
                           :expected expected-position :actual position})))
        (when-not (< (long (:token-count route))
                     (long (:maxpos (:decode-state decoder))))
          (throw (ex-info "Paged decode reached its route capacity"
                          {:continuation-id continuation-id
                           :token-count (:token-count route)
                           :capacity (:maxpos (:decode-state decoder))})))))
    (let [batch (paged-append/reserve-batch! (:pool decoder) continuation-ids)]
      (try
        (let [entries (paged-append/reservation-entries batch)
              lease (page-pool/acquire-prospective-lease! (:pool decoder) entries)]
          (try
            (let [compact-route-values
                  (page-pool/leased-dense-route-values
                   (:pool decoder) lease
                   {:pages-per-sequence (:pages-per-sequence decoder)})
                  route-values
                  (scatter-route-values batch-size (:pages-per-sequence decoder)
                                        lanes compact-route-values)
                  slots (int-array batch-size -1)
                  positions (int-array batch-size)
                  compact-slots ^ints (paged-append/slot-values batch)
                  keys (:descriptor-keys decoder)]
              (doseq [[source-row {:keys [lane position]}]
                      (map-indexed vector lane-work)]
                (aset slots (int lane) (aget compact-slots source-row))
                (aset positions (int lane) (int position)))
              (gpu/upload-ranges!
               sess
               [[(:slots keys) slots {:elements batch-size}]
                [(:row-offsets keys) (int-array (range (inc batch-size)))
                 {:elements (inc batch-size)}]
                [(:positions keys) positions {:elements batch-size}]
                [(:page-table keys) (:page-table route-values)
                 {:elements (alength ^ints (:page-table route-values))}]
                [(:lengths keys) (:lengths route-values)
                 {:elements batch-size}]
                [(:start-positions keys) (:start-positions route-values)
                 {:elements batch-size}]])
              (gpu-link/run! (:executable decoder)))
            (finally
              (page-pool/release-lease! (:pool decoder) lease))))
        (paged-append/commit-batch! batch)
        (let [tokens ^ints (gpu/download sess :tokbuf)]
          (mapv #(assoc % :token (long (aget tokens (int (:lane %))))) lane-work))
        (catch Throwable error
          (when (= :reserved @(:state batch))
            (paged-append/abort-batch! batch))
          (throw error))))))

(defn step-batch!
  "Process one resident token for every continuation in a fixed decode batch.

  `continuation-ids` and `positions` must each have the decoder's bound batch
  size and retain lane order. Every position must extend its route. The complete
  batch reserves, executes, and publishes atomically with respect to logical
  route lengths; on failure all still-pending reservations are aborted. Returns
  the next greedy token id for each lane in the same order."
  [decoder continuation-ids positions]
  (let [{:keys [batch-size]} (:decode-state decoder)
        batch-size (long (or batch-size 1))
        continuation-ids (vec continuation-ids)
        positions (mapv long positions)]
    (when-not (and (= batch-size (count continuation-ids))
                   (= batch-size (count positions)))
      (throw (ex-info "Paged step does not match the bound decode batch"
                      {:batch-size batch-size
                       :continuation-count (count continuation-ids)
                       :position-count (count positions)})))
    (mapv :token
          (step-lanes!
           decoder
           (mapv (fn [lane continuation-id position]
                   {:lane lane :continuation-id continuation-id :position position})
                 (range batch-size) continuation-ids positions)))))

(defn step!
  "Process the resident input token and append one K/V row atomically.

  `position` must be the route's next absolute position. Every layer writes the
  reserved physical slot and consumes the prospective route before the logical
  token count is published. On failure the reservation is rolled back and its
  partially written slot remains unreachable. Returns the next greedy token id."
  [decoder continuation-id position]
  (let [batch-size (long (:batch-size (:decode-state decoder) 1))]
    (when-not (= batch-size 1)
      (throw (ex-info "Use step-batch! with a batched paged decoder"
                      {:batch-size batch-size})))
    (first (step-batch! decoder [continuation-id] [position]))))

(defn decode-token!
  "Upload `token`'s embedding, process it at `position`, and return the next token."
  [decoder continuation-id token position]
  (prime-token! decoder token)
  (step! decoder continuation-id position))

(defn decode-tokens!
  "Upload one token per lane, process the fixed continuation batch, and return
  the lane-ordered next tokens."
  [decoder continuation-ids tokens positions]
  (prime-tokens! decoder tokens)
  (step-batch! decoder continuation-ids positions))

(defn prefill-range!
  "Process one complete multi-row prompt tile for a resident continuation.

  `tokens` must contain exactly the decoder's configured `:prefill-T` rows and
  `start-position` must extend the route. All K/V rows remain unpublished until
  every layer has completed; failures roll the whole range reservation back.
  Returns `decoder` without changing the resident one-row decode input."
  [decoder continuation-id tokens start-position]
  (require-open! decoder)
  (let [{:keys [sess model maxpos batch-size]} (:decode-state decoder)
        tile-size (some-> (:prefill-T decoder) long)
        tokens (vec tokens)
        start-position (long start-position)
        route (or (page-pool/route (:pool decoder) continuation-id)
                  (throw (ex-info "Continuation is not resident"
                                  {:continuation-id continuation-id})))
        expected-position (+ (long (:start-position route))
                             (long (:token-count route)))]
    (when-not (and tile-size (:prefill-executable decoder))
      (throw (ex-info "Paged decoder has no resident bulk-prefill executable" {})))
    (when-not (= 1 (long (or batch-size 1)))
      (throw (ex-info "Bulk prompt prefill currently supports one sequence"
                      {:batch-size batch-size})))
    (when-not (= tile-size (count tokens))
      (throw (ex-info "Prompt tile does not match the bound prefill row count"
                      {:prefill-T tile-size :token-count (count tokens)})))
    (when-not (= expected-position start-position)
      (throw (ex-info "Paged prefill range does not extend the resident route"
                      {:continuation-id continuation-id
                       :expected expected-position :actual start-position})))
    (when (> (+ (long (:token-count route)) tile-size) (long maxpos))
      (throw (ex-info "Paged prefill range exceeds route capacity"
                      {:continuation-id continuation-id
                       :token-count (:token-count route)
                       :prefill-T tile-size
                       :capacity maxpos})))
    (let [batch (paged-append/reserve-range! (:pool decoder) continuation-id tile-size)]
      (try
        (let [entries (paged-append/reservation-entries batch)
              lease (page-pool/acquire-prospective-lease! (:pool decoder) entries)]
          (try
            (let [route-values
                  (page-pool/leased-dense-route-values
                   (:pool decoder) lease
                   {:pages-per-sequence (:pages-per-sequence decoder)})
                  d (long (:d-model model))
                  ^floats rows (float-array (* tile-size d))
                  positions (int-array (map int (range start-position
                                                        (+ start-position tile-size))))
                  keys (:prefill-descriptor-keys decoder)]
              (doseq [[row token] (map-indexed vector tokens)]
                (System/arraycopy ^floats (decoder-gpu/embed-row model token) 0
                                  rows (int (* (long row) d)) (int d)))
              (gpu/upload-ranges!
               sess
               [[:pr0 rows {:elements (alength rows)}]
                [(:slots keys) (paged-append/slot-values batch)
                 {:elements tile-size}]
                [(:row-offsets keys) (int-array [0 tile-size]) {:elements 2}]
                [(:positions keys) positions {:elements tile-size}]
                [(:page-table keys) (:page-table route-values)
                 {:elements (alength ^ints (:page-table route-values))}]
                [(:lengths keys) (:lengths route-values) {:elements 1}]
                [(:start-positions keys) (:start-positions route-values)
                 {:elements 1}]])
              (gpu-link/run! (:prefill-executable decoder)))
            (finally
              (page-pool/release-lease! (:pool decoder) lease))))
        (paged-append/commit-batch! batch)
        decoder
        (catch Throwable error
          (when (= :reserved @(:state batch))
            (paged-append/abort-batch! batch))
          (throw error))))))

(defn prime-prompts-batch!
  "Compute an equal-length missing suffix for one prompt per decode lane.

  Routes may already contain different exact prefix lengths, but every lane
  must require the same number of decode steps so the fixed graph stays full.
  The final prompt token in each lane is left resident and pending."
  [decoder continuation-ids prompts]
  (require-open! decoder)
  (let [continuation-ids (vec continuation-ids)
        prompts (mapv vec prompts)
        batch-size (long (:batch-size (:decode-state decoder) 1))]
    (when-not (and (= batch-size (count continuation-ids))
                   (= batch-size (count prompts))
                   (every? seq prompts))
      (throw (ex-info "Prompt batch does not match the bound decoder"
                      {:batch-size batch-size
                       :continuation-count (count continuation-ids)
                       :prompt-count (count prompts)})))
    (let [routes (mapv #(or (page-pool/route (:pool decoder) %)
                            (allocate-continuation! decoder %))
                       continuation-ids)
          cached-counts (mapv (comp long :token-count) routes)
          processed-counts (mapv (comp dec count) prompts)
          missing-counts (mapv - processed-counts cached-counts)]
      (when-not (every? zero? (map (comp long :start-position) routes))
        (throw (ex-info "Paged prompt priming requires zero absolute starts"
                        {:start-positions (mapv :start-position routes)})))
      (when (some neg? missing-counts)
        (throw (ex-info "A resident prefix is longer than its prompt"
                        {:cached-counts cached-counts
                         :processed-counts processed-counts})))
      (when-not (apply = missing-counts)
        (throw (ex-info "Fixed decode lanes require equal missing suffix lengths"
                        {:missing-counts missing-counts})))
      (dotimes [offset (long (first missing-counts))]
        (let [positions (mapv #(+ % offset) cached-counts)
              tokens (mapv #(nth %1 %2) prompts positions)]
          (decode-tokens! decoder continuation-ids tokens positions)))
      (prime-tokens! decoder (mapv peek prompts)))))

(defn generate-batch!
  "Greedily generate exactly `max-new` tokens per fixed decode lane.

  Prompts first pass through `prime-prompts-batch!`; consequently their missing
  suffix lengths must match. Generation stops for the whole batch when any lane
  reaches capacity. Per-lane EOS retirement requires a scheduler with lane
  refill and is intentionally not simulated by advancing a finished route."
  [decoder continuation-ids prompts max-new]
  (let [continuation-ids (vec continuation-ids)
        prompts (mapv vec prompts)]
    (prime-prompts-batch! decoder continuation-ids prompts)
    (loop [positions (mapv (comp dec count) prompts)
           output (vec (repeat (count prompts) []))]
      (if (and (< (count (first output)) (long max-new))
               (every? #(< (long (:token-count (page-pool/route (:pool decoder) %)))
                            (long (:maxpos (:decode-state decoder))))
                       continuation-ids))
        (let [tokens (step-batch! decoder continuation-ids positions)]
          (recur (mapv inc positions) (mapv conj output tokens)))
        output))))

(defn prime-prompt!
  "Compute only the uncached suffix of `prompt` and leave its final token pending.

  `continuation-id` may name an empty route or a restored exact prefix. Its
  token count must not exceed `dec(count(prompt))`, and prompt routes currently
  start at absolute position zero. Existing page rows are never recomputed.
  Returns the decoder after priming the pending token's resident embedding."
  [decoder continuation-id prompt]
  (require-open! decoder)
  (when-not (seq prompt)
    (throw (ex-info "Paged prompt priming requires a nonempty prompt" {})))
  (let [resident-route
        (or (page-pool/route (:pool decoder) continuation-id)
            (allocate-continuation! decoder continuation-id))
        cached-count (long (:token-count resident-route))
        processed-count (dec (count prompt))]
    (when-not (zero? (long (:start-position resident-route)))
      (throw (ex-info "Paged prompt priming requires a zero absolute start position"
                      {:continuation-id continuation-id
                       :start-position (:start-position resident-route)})))
    (when (> cached-count processed-count)
      (throw (ex-info "Resident prefix is longer than the prompt's processed prefix"
                      {:continuation-id continuation-id
                       :cached-token-count cached-count
                       :processed-token-count processed-count})))
    (let [prompt (vec prompt)
          tile-size (some-> (:prefill-T decoder) long)
          bulk-end (if tile-size
                     (+ cached-count
                        (* tile-size (quot (- processed-count cached-count)
                                           tile-size)))
                     cached-count)]
      (when tile-size
        (doseq [position (range cached-count bulk-end tile-size)]
          (prefill-range! decoder continuation-id
                          (subvec prompt position (+ position tile-size))
                          position)))
      (doseq [position (range bulk-end processed-count)]
        (decode-token! decoder continuation-id (nth prompt position) position)))
    (prime-token! decoder (last prompt))))

(defn generate!
  "Greedily generate up to `max-new` ids using a paged continuation route.

  The route may be empty or contain an exact restored prefix. Only the missing
  prompt suffix is committed to K/V; the last token is then processed by the
  rollout. Generation stops at an id in `eos-ids` or at the decode state's
  maximum position."
  [decoder continuation-id prompt max-new & {:keys [eos-ids]
                                             :or {eos-ids #{}}}]
  (require-open! decoder)
  (when-not (seq prompt)
    (throw (ex-info "Paged generation requires a nonempty prompt" {})))
  (let [prefix-count (dec (count prompt))]
    (prime-prompt! decoder continuation-id prompt)
    (loop [position prefix-count
             output []]
      (if (and (< (count output) (long max-new))
               (< (:token-count (page-pool/route (:pool decoder) continuation-id))
                  (long (:maxpos (:decode-state decoder)))))
        (let [token (step! decoder continuation-id position)
              output (conj output token)]
          (if (contains? eos-ids token)
            output
            (recur (inc position) output)))
        output))))

(defn close!
  "Release the composite paged executable and its route descriptors. Idempotent;
  does not close the decode state's staged LinkPlans, page pool, or session."
  [decoder]
  (locking decoder
    (when-not (:closed? @(:state decoder))
      (gpu-link/close! (:executable decoder))
      (when-let [prefill-executable (:prefill-executable decoder)]
        (gpu-link/close! prefill-executable))
      (doseq [key (:owned-buffer-keys @(:state decoder))]
        (when (gpu/buffer (get-in decoder [:decode-state :sess]) key)
          (gpu/free-buffer! (get-in decoder [:decode-state :sess]) key)))
      (swap! (:state decoder) assoc :closed? true)))
  nil)
