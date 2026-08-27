(ns pretrained.continuation.page-pool
  "Worker-local paged resident attention state.

  One logical physical page spans every declared attention-state slab and layer.
  Continuations therefore share, copy, and release a coherent model state page,
  while Raster owns the stable device allocations and checked buffer views."
  (:require [pretrained.attention-state :as attention-state]
            [raster.gpu.core :as gpu])
  (:import [java.lang.foreign MemorySegment]
           [java.util UUID]))

(defrecord DevicePagePool
           [session layout page-size physical-pages dtype buffer-keys state])

(defrecord PageLease [pool id routes])

(defn page-pool?
  "Return true when `value` is a device page pool."
  [value]
  (instance? DevicePagePool value))

(defn page-lease?
  "Return true when `value` is a resident page lease."
  [value]
  (instance? PageLease value))

(defn- canonical-dtype
  [dtype]
  (case dtype
    (:float16 :f16) :half
    (:float32 :f32) :float
    dtype))

(defn- checked-positive
  [field value]
  (when-not (and (integer? value) (pos? value))
    (throw (ex-info "Page-pool extent must be a positive integer"
                    {:field field :value value})))
  (long value))

(defn- checked-product
  [field values]
  (try
    (reduce #(Math/multiplyExact (long %1) (long %2)) 1 values)
    (catch ArithmeticException error
      (throw (ex-info "Page-pool extent exceeds signed 64-bit capacity"
                      {:field field :values (vec values)} error)))))

(defn- page-count
  [token-count page-size]
  (if (zero? token-count)
    0
    (inc (quot (dec token-count) page-size))))

(defn- transfer-runs
  [resident-route start token-count page-size]
  (loop [relative-token 0
         runs []]
    (if (= relative-token token-count)
      runs
      (let [logical-token (+ start relative-token)
            logical-page (quot logical-token page-size)
            offset-in-page (rem logical-token page-size)
            physical-page (nth (:pages resident-route) logical-page)
            run-tokens (min (- token-count relative-token)
                            (- page-size offset-in-page))
            physical-token (+ (* physical-page page-size) offset-in-page)
            previous (peek runs)]
        (if (and previous
                 (= physical-token
                    (+ (:physical-token previous) (:token-count previous))))
          (recur (+ relative-token run-tokens)
                 (update-in runs [(dec (count runs)) :token-count] + run-tokens))
          (recur (+ relative-token run-tokens)
                 (conj runs {:relative-token relative-token
                             :physical-token physical-token
                             :token-count run-tokens})))))))

(defn- record-transfer!
  [pool measurement]
  (let [counter-key [(:direction measurement)
                     (:timing-source measurement)
                     (boolean (:asynchronous? measurement))]]
    (swap! (:state pool)
           (fn [state]
             (-> state
                 (assoc-in [:transfers :last] measurement)
                 (update-in [:transfers :counters counter-key :submissions]
                            (fnil inc 0))
                 (update-in [:transfers :counters counter-key :bytes]
                            (fnil + 0) (long (:bytes measurement 0)))
                 (update-in [:transfers :counters counter-key :commands]
                            (fnil + 0) (long (:commands measurement 0)))
                 (update-in [:transfers :counters counter-key :elapsed-ns]
                            (fnil + 0) (long (:elapsed-ns measurement 0)))
                 (update-in [:transfers :counters counter-key :submit-host-ns]
                            (fnil + 0) (long (:submit-host-ns measurement 0)))
                 (update-in [:transfers :counters counter-key :host-wall-ns]
                            (fnil + 0) (long (:host-wall-ns measurement 0))))))))

(defn- transfer-ranges-measured!
  [pool direction entries]
  (let [session (:session pool)
        event ((case direction
                 :upload gpu/submit-upload-ranges!
                 :download gpu/submit-download-ranges!)
               session entries)]
    (try
      (let [value (gpu/await-event! session event)
            measurement (gpu/event-measurement session event)]
        (record-transfer! pool measurement)
        value)
      (finally
        (gpu/release-event! session event)))))

(defn- buffer-key
  [prefix slab layer]
  (keyword (str prefix "-" (name (:name slab)) "-" (long layer))))

(defn open-pool!
  "Allocate stable resident page pools for an attention-state layout.

  `opts` requires positive `:page-size` and `:physical-pages`. `:dtype` defaults
  to FP16, the current Raster routed-attention storage format. `:key-prefix` may
  be supplied when deterministic session buffer names are useful. Returns a
  `DevicePagePool`; callers must close the owning Raster session.

  Throws when the layout is malformed, the dtype is unsupported, or allocation
  fails. Allocation rollback is owned by Raster's session lifecycle."
  [session layout {:keys [page-size physical-pages dtype key-prefix]
                   :or {dtype :half}}]
  (let [page-size (checked-positive :page-size page-size)
        physical-pages (checked-positive :physical-pages physical-pages)
        dtype (canonical-dtype dtype)
        prefix (or key-prefix (str "paged-" (UUID/randomUUID)))
        slabs (:slabs layout)]
    (when-not (and (map? layout) (seq slabs))
      (throw (ex-info "Page pool requires a resolved attention-state layout"
                      {:layout layout})))
    (when (or (> page-size Integer/MAX_VALUE)
              (> physical-pages Integer/MAX_VALUE))
      (throw (ex-info "Raster page routing uses int32 page coordinates"
                      {:page-size page-size :physical-pages physical-pages})))
    (when-not (contains? #{:half :float} dtype)
      (throw (ex-info "Page pool currently supports FP16 or FP32 storage"
                      {:dtype dtype})))
    (doseq [slab slabs]
      (when-not (and (keyword? (:name slab))
                     (integer? (:count slab)) (pos? (:count slab))
                     (integer? (:elements-per-token slab))
                     (pos? (:elements-per-token slab)))
        (throw (ex-info "Page pool slab layout is incomplete"
                        {:slab slab}))))
    (when-not (= (count slabs) (count (set (map :name slabs))))
      (throw (ex-info "Page pool slab names must be unique"
                      {:slab-names (mapv :name slabs)})))
    (let [keys (into {}
                     (for [slab slabs
                           layer (range (:count slab))]
                       [[(:name slab) layer] (buffer-key prefix slab layer)]))
          specs (into {}
                      (for [slab slabs
                            layer (range (:count slab))
                            :let [elements (checked-product
                                            :slab-pool
                                            [physical-pages page-size
                                             (:elements-per-token slab)])]]
                        [(get keys [(:name slab) layer])
                         [dtype elements nil :state]]))]
      (gpu/alloc! session specs)
      (->DevicePagePool
       session layout page-size physical-pages dtype keys
       (atom {:free (apply sorted-set (range physical-pages))
              :refcounts {}
              :leases {}
              :routes {}
              :transfers {:counters {} :last nil}})))))

(defn buffer-keys
  "Return `{[slab-name layer] session-buffer-key}` for `pool`."
  [pool]
  (:buffer-keys pool))

(defn route
  "Return a continuation's resident route, or nil when it is not resident."
  [pool continuation-id]
  (get-in @(:state pool) [:routes continuation-id]))

(defn residency-snapshot
  "Return immutable route, reference, lease, and capacity data for policy decisions.

  The result contains no mutable pool state. It is intended for worker-local
  admission and eviction planners; callers must still revalidate a plan while
  applying it because GPU work may acquire a lease concurrently."
  [pool]
  (locking pool
    (let [{:keys [free refcounts routes leases]} @(:state pool)]
      {:physical-pages (:physical-pages pool)
       :page-size (:page-size pool)
       :free-pages (set free)
       :refcounts refcounts
       :routes routes
       :leases (or leases {})})))

(defn touch-route!
  "Record worker-local policy observations for a resident continuation.

  `observations` is merged into the route's `:cache/policy` map. Policy metadata
  never changes page identity or tensor contents. Returns the updated route and
  throws when the continuation is not resident."
  [pool continuation-id observations]
  (when-not (map? observations)
    (throw (ex-info "Route policy observations must be a map"
                    {:continuation-id continuation-id
                     :observations observations})))
  (locking pool
    (let [state @(:state pool)
          resident-route (get-in state [:routes continuation-id])]
      (when-not resident-route
        (throw (ex-info "Cannot touch a nonresident continuation"
                        {:continuation-id continuation-id})))
      (let [updated (update resident-route :cache/policy
                            #(merge (or % {}) observations))]
        (reset! (:state pool) (assoc-in state [:routes continuation-id] updated))
        updated))))

(defn free-page-count
  "Return the number of currently unreferenced physical pages."
  [pool]
  (count (:free @(:state pool))))

(defn stats
  "Return instantaneous worker-local page ownership and capacity counters."
  [pool]
  (let [{:keys [free refcounts routes leases]} @(:state pool)]
    {:physical-pages (:physical-pages pool)
     :free-pages (count free)
     :resident-pages (count refcounts)
     :shared-pages (count (filter #(> % 1) (vals refcounts)))
     :resident-routes (count routes)
     :active-leases (count leases)}))

(defn transfer-stats
  "Return cumulative measured cache-transfer counters and the last measurement.

  Counters are keyed by `[direction timing-source asynchronous?]`, preserving
  the distinction between device-event timings and inline host-coherent copies.
  Values are immutable snapshots suitable for metrics and benchmark reports."
  [pool]
  (or (:transfers @(:state pool)) {:counters {} :last nil}))

(defn- take-free-pages
  [state n continuation-id]
  (let [pages (vec (take n (:free state)))]
    (when-not (= n (count pages))
      (throw (ex-info "Device page pool has insufficient free capacity"
                      {:continuation-id continuation-id
                       :required n :available (count (:free state))})))
    [pages (-> state
               (update :free #(apply disj % pages))
               (update :refcounts
                       #(reduce (fn [references page] (assoc references page 1))
                                % pages)))]))

(defn allocate-route!
  "Allocate a resident page route for `token-count` logical tokens.

  Pages are initially uninitialized; callers normally follow this with
  `restore-chunk!` or token appends. `:start-position` defaults to zero and
  records the absolute position of routed token zero."
  ([pool continuation-id token-count]
   (allocate-route! pool continuation-id token-count {}))
  ([pool continuation-id token-count {:keys [start-position policy]
                                      :or {start-position 0 policy {}}}]
   (let [token-count (long token-count)
         start-position (long start-position)]
     (when (or (nil? continuation-id) (neg? token-count) (neg? start-position))
       (throw (ex-info "Resident route identity and extents are invalid"
                       {:continuation-id continuation-id
                        :token-count token-count
                        :start-position start-position})))
     (when-not (map? policy)
       (throw (ex-info "Resident route policy must be a map"
                       {:continuation-id continuation-id :policy policy})))
     (locking pool
       (let [state @(:state pool)]
         (when (get-in state [:routes continuation-id])
           (throw (ex-info "Continuation already has a resident route"
                           {:continuation-id continuation-id})))
         (let [required (page-count token-count (:page-size pool))
               [pages next-state] (take-free-pages state required continuation-id)
               resident-route (cond-> {:continuation-id continuation-id
                                       :pages pages
                                       :token-count token-count
                                       :start-position start-position}
                                (seq policy) (assoc :cache/policy policy))]
           (reset! (:state pool)
                   (assoc-in next-state [:routes continuation-id] resident-route))
           resident-route))))))

(defn fork-route!
  "Create `target-id` as a page-sharing snapshot of `source-id`.

  A later append to a shared partial tail performs copy-on-write. Full shared
  pages remain immutable and can be batched through independent page tables."
  [pool source-id target-id]
  (locking pool
    (let [state @(:state pool)
          source (get-in state [:routes source-id])]
      (when (nil? target-id)
        (throw (ex-info "Target continuation requires an identity" {})))
      (when-not source
        (throw (ex-info "Source continuation is not resident"
                        {:continuation-id source-id})))
      (when (get-in state [:routes target-id])
        (throw (ex-info "Target continuation already has a resident route"
                        {:continuation-id target-id})))
      (when (:pending source)
        (throw (ex-info "Cannot fork a continuation with an uncommitted append"
                        {:continuation-id source-id})))
      (let [target (assoc source :continuation-id target-id)
            next-state (-> state
                           (update :refcounts
                                   #(reduce (fn [refs page] (update refs page inc))
                                            % (:pages source)))
                           (assoc-in [:routes target-id] target))]
        (reset! (:state pool) next-state)
        target))))

(defn- release-page
  [state page]
  (let [refs (get-in state [:refcounts page] 0)]
    (when-not (pos? refs)
      (throw (ex-info "Resident page reference count is corrupt"
                      {:page page :refcount refs})))
    (if (= refs 1)
      (-> state
          (update :refcounts dissoc page)
          (update :free conj page))
      (update-in state [:refcounts page] dec))))

(defn release-route!
  "Release a continuation route and return true, or false when it was absent."
  [pool continuation-id]
  (locking pool
    (let [state @(:state pool)
          resident-route (get-in state [:routes continuation-id])]
      (if-not resident-route
        false
        (do
          (when (:pending resident-route)
            (throw (ex-info "Cannot release a route with an uncommitted append"
                            {:continuation-id continuation-id})))
          (reset! (:state pool)
                  (reduce release-page
                          (update state :routes dissoc continuation-id)
                          (:pages resident-route)))
          true)))))

(defn acquire-lease!
  "Pin immutable route snapshots for one scheduled batch.

  Every occurrence in `continuation-ids` contributes a reference, matching its
  page-table row. A concurrent append will consequently copy a shared partial
  tail rather than modifying the leased snapshot. Returns a `PageLease` that
  must be released after GPU completion."
  [pool continuation-ids]
  (let [continuation-ids (vec continuation-ids)]
    (when-not (seq continuation-ids)
      (throw (ex-info "A page lease requires at least one continuation" {})))
    (locking pool
      (let [state @(:state pool)
            routes (mapv #(or (get-in state [:routes %])
                              (throw (ex-info "Cannot lease a nonresident continuation"
                                              {:continuation-id %})))
                         continuation-ids)]
        (when-let [pending (some #(when (:pending %) (:continuation-id %)) routes)]
          (throw (ex-info "Cannot lease a route with an uncommitted append"
                          {:continuation-id pending})))
        (let [id (UUID/randomUUID)
              pages (vec (mapcat :pages routes))
              lease (->PageLease pool id routes)
              next-state (-> state
                             (update :refcounts
                                     #(reduce (fn [references page]
                                                (update references page (fnil inc 0)))
                                              % pages))
                             (assoc-in [:leases id]
                                       {:pages pages
                                        :continuation-ids continuation-ids}))]
          (reset! (:state pool) next-state)
          lease)))))

(defn acquire-prospective-lease!
  "Pin the post-append route snapshots named by `reservation-entries`.

  Each entry is `{:continuation-id id :reservation reservation}` and must match
  the route's current pending append exactly. The leased snapshot includes the
  reserved page and increments its token count, while the live route remains
  uncommitted. This lets an in-order GPU queue submit append followed by
  attention without making unfinished state visible to other schedulers."
  [pool reservation-entries]
  (let [entries (vec reservation-entries)
        continuation-ids (mapv :continuation-id entries)]
    (when-not (seq entries)
      (throw (ex-info "A prospective page lease requires append reservations" {})))
    (when-not (= (count continuation-ids) (count (set continuation-ids)))
      (throw (ex-info "Prospective lease continuation identities must be unique"
                      {:continuation-ids continuation-ids})))
    (locking pool
      (let [state @(:state pool)
            routes
            (mapv
             (fn [{:keys [continuation-id reservation]}]
               (let [resident-route (get-in state [:routes continuation-id])]
                 (when-not resident-route
                   (throw (ex-info "Cannot lease a nonresident continuation"
                                   {:continuation-id continuation-id})))
                 (when-not (and reservation
                                (= reservation (:pending resident-route)))
                   (throw (ex-info "Prospective lease reservation is stale"
                                   {:continuation-id continuation-id
                                    :expected (:pending resident-route)
                                    :actual reservation})))
                 (-> resident-route
                     (dissoc :pending)
                     (update :token-count inc))))
             entries)
            id (UUID/randomUUID)
            pages (vec (mapcat :pages routes))
            lease (->PageLease pool id routes)
            next-state (-> state
                           (update :refcounts
                                   #(reduce (fn [references page]
                                              (update references page (fnil inc 0)))
                                            % pages))
                           (assoc-in [:leases id]
                                     {:pages pages
                                      :continuation-ids continuation-ids
                                      :prospective? true}))]
        (reset! (:state pool) next-state)
        lease))))

(defn release-lease!
  "Release a page lease and return true, or false when already released."
  [pool lease]
  (when-not (and (page-lease? lease) (identical? pool (:pool lease)))
    (throw (ex-info "Page lease belongs to a different pool"
                    {:lease lease})))
  (locking pool
    (let [state @(:state pool)
          entry (get-in state [:leases (:id lease)])]
      (if-not entry
        false
        (do
          (reset! (:state pool)
                  (reduce release-page
                          (update state :leases dissoc (:id lease))
                          (:pages entry)))
          true)))))

(defn page-view
  "Return a stable Raster view of one slab/layer physical page."
  [pool slab-name layer physical-page]
  (let [slab (some #(when (= slab-name (:name %)) %) (:slabs (:layout pool)))
        key (get (:buffer-keys pool) [slab-name layer])]
    (when-not (and slab key (<= 0 (long physical-page))
                   (< (long physical-page) (:physical-pages pool)))
      (throw (ex-info "Page view does not name a pool slab, layer, and page"
                      {:slab slab-name :layer layer :page physical-page})))
    (when-not (< (long layer) (:count slab))
      (throw (ex-info "Page view layer is outside its slab"
                      {:slab slab-name :layer layer :layers (:count slab)})))
    (let [elements (checked-product :page [(:page-size pool)
                                            (:elements-per-token slab)])
          element-bytes (case (:dtype pool) :half 2 :float 4)]
      (gpu/buffer-view (:session pool) key
                       {:byte-offset (* (long physical-page) elements element-bytes)
                        :shape [elements]
                        :id [key physical-page]}))))

(defn- token-range-view
  [pool slab layer physical-token token-count]
  (let [key (get (:buffer-keys pool) [(:name slab) layer])
        per-token (:elements-per-token slab)
        elements (checked-product :token-range [token-count per-token])
        element-offset (checked-product :token-range-offset
                                        [physical-token per-token])
        element-bytes (case (:dtype pool) :half 2 :float 4)]
    (gpu/buffer-view (:session pool) key
                     {:byte-offset (* element-offset element-bytes)
                      :shape [elements]
                      :id [key physical-token token-count]})))

(defn- copy-page!
  [pool source-page destination-page]
  (doseq [slab (:slabs (:layout pool))
          layer (range (:count slab))
          :let [elements (* (:page-size pool) (:elements-per-token slab))]]
    (gpu/copy-range! (:session pool)
                     (page-view pool (:name slab) layer source-page)
                     (page-view pool (:name slab) layer destination-page)
                     {:elements elements})))

(defn reserve-append!
  "Reserve the physical slot for one append and return its page/offset.

  A shared partial tail is copied before the route changes. The reservation must
  be completed with `commit-append!` or reverted with `abort-append!`. Only one
  append may be pending per continuation."
  [pool continuation-id]
  (locking pool
    (let [state @(:state pool)
          resident-route (get-in state [:routes continuation-id])]
      (when-not resident-route
        (throw (ex-info "Continuation is not resident"
                        {:continuation-id continuation-id})))
      (when (:pending resident-route)
        (throw (ex-info "Continuation already has a pending append"
                        {:continuation-id continuation-id})))
      (let [token-count (:token-count resident-route)
            logical-page (quot token-count (:page-size pool))
            page-offset (rem token-count (:page-size pool))
            existing (get (:pages resident-route) logical-page)
            shared? (and existing (> (get-in state [:refcounts existing]) 1))
            needs-page? (or (nil? existing) shared?)
            [new-pages reserved-state]
            (if needs-page?
              (take-free-pages state 1 continuation-id)
              [[] state])
            physical-page (or (first new-pages) existing)]
        (when shared?
          (copy-page! pool existing physical-page))
        (let [pages (cond
                      (nil? existing) (conj (:pages resident-route) physical-page)
                      shared? (assoc (:pages resident-route) logical-page physical-page)
                      :else (:pages resident-route))
              pending {:expected-token-count token-count
                       :logical-page logical-page
                       :physical-page physical-page
                       :page-offset page-offset
                       :added-page? (nil? existing)
                       :replaced-page (when shared? existing)}
              next-state (cond-> reserved-state
                           shared? (release-page existing)
                           true (assoc-in [:routes continuation-id]
                                          (assoc resident-route
                                                 :pages pages
                                                 :pending pending)))]
          (reset! (:state pool) next-state)
          pending)))))

(defn- validate-reservation-entries!
  [state reservation-entries]
  (let [entries (vec reservation-entries)
        continuation-ids (mapv :continuation-id entries)]
    (when-not (seq entries)
      (throw (ex-info "Append completion requires at least one reservation" {})))
    (when-not (= (count continuation-ids) (count (set continuation-ids)))
      (throw (ex-info "Append completion continuation identities must be unique"
                      {:continuation-ids continuation-ids})))
    (doseq [{:keys [continuation-id reservation]} entries]
      (let [resident-route (get-in state [:routes continuation-id])
            pending (:pending resident-route)]
        (when-not resident-route
          (throw (ex-info "Append reservation continuation is not resident"
                          {:continuation-id continuation-id})))
        (when-not (and reservation (= reservation pending))
          (throw (ex-info "Append reservation is stale or belongs to another route"
                          {:continuation-id continuation-id
                           :expected pending :actual reservation})))))
    entries))

(defn commit-appends!
  "Atomically commit a batch of successful append reservations.

  `reservation-entries` contains maps with `:continuation-id` and
  `:reservation`. Every reservation is revalidated before any route changes.
  Returns the updated routes in entry order."
  [pool reservation-entries]
  (locking pool
    (let [state @(:state pool)
          entries (validate-reservation-entries! state reservation-entries)
          next-state
          (reduce
           (fn [current {:keys [continuation-id]}]
             (update-in current [:routes continuation-id]
                        #(-> % (dissoc :pending) (update :token-count inc))))
           state entries)]
      (reset! (:state pool) next-state)
      (mapv #(get-in next-state [:routes (:continuation-id %)]) entries))))

(defn commit-append!
  "Commit a successful append reservation and return the updated route."
  [pool continuation-id reservation]
  (first (commit-appends!
          pool [{:continuation-id continuation-id :reservation reservation}])))

(defn- abort-reservation
  [state {:keys [continuation-id]}]
  (let [resident-route (get-in state [:routes continuation-id])
        {:keys [logical-page physical-page added-page? replaced-page]}
        (:pending resident-route)
        pages (cond
                added-page? (pop (:pages resident-route))
                replaced-page (assoc (:pages resident-route) logical-page replaced-page)
                :else (:pages resident-route))]
    (cond-> state
      (or added-page? replaced-page) (release-page physical-page)
      replaced-page (update-in [:refcounts replaced-page] (fnil inc 0))
      true (assoc-in [:routes continuation-id]
                     (assoc (dissoc resident-route :pending) :pages pages)))))

(defn abort-appends!
  "Atomically revert a batch of append reservations.

  Every reservation is revalidated before any allocation or route metadata is
  changed. Returns the restored routes in entry order."
  [pool reservation-entries]
  (locking pool
    (let [state @(:state pool)
          entries (validate-reservation-entries! state reservation-entries)
          next-state (reduce abort-reservation state entries)]
      (reset! (:state pool) next-state)
      (mapv #(get-in next-state [:routes (:continuation-id %)]) entries))))

(defn abort-append!
  "Revert an append reservation and return the unchanged logical route."
  [pool continuation-id reservation]
  (first (abort-appends!
          pool [{:continuation-id continuation-id :reservation reservation}])))

(defn- payload-elements
  [payload]
  (cond
    (instance? (Class/forName "[F") payload) (alength ^floats payload)
    (instance? (Class/forName "[S") payload) (alength ^shorts payload)
    (instance? MemorySegment payload) nil
    :else (throw (ex-info "Attention-state payload has an unsupported representation"
                          {:payload-type (type payload)}))))

(defn- checked-fp16-segment
  [^MemorySegment segment elements]
  (when-not (= (.byteSize segment) (* 2 elements))
    (throw (ex-info "Mapped FP16 payload length differs from its descriptor"
                    {:expected-bytes (* 2 elements) :actual-bytes (.byteSize segment)})))
  segment)

(defn- half-payload
  [payload source-dtype elements]
  (let [source-dtype (canonical-dtype source-dtype)]
    (cond
      (and (= :half source-dtype) (instance? (Class/forName "[S") payload))
      (do
        (when-not (= elements (alength ^shorts payload))
          (throw (ex-info "FP16 payload length differs from its descriptor"
                          {:expected elements :actual (alength ^shorts payload)})))
        payload)

      (and (= :half source-dtype) (instance? MemorySegment payload))
      (checked-fp16-segment payload elements)

      (and (= :float source-dtype) (instance? (Class/forName "[F") payload))
      (do
        (when-not (= elements (alength ^floats payload))
          (throw (ex-info "FP32 payload length differs from its descriptor"
                          {:expected elements :actual (alength ^floats payload)})))
        (short-array (map #(Float/floatToFloat16 (float %)) payload)))

      :else
      (throw (ex-info "Payload dtype and representation cannot be restored to FP16"
                      {:source-dtype source-dtype :payload-type (type payload)
                       :payload-elements (payload-elements payload)})))))

(defn restore-chunk!
  "Scatter one durable chunk payload into an allocated resident route.

  The chunk may begin/end inside pages and cross arbitrary physical-page IDs.
  Current Raster attention consumes FP16 pools. FP16 mmap segments upload
  directly without an intermediate JVM array. Returns the resident route."
  [pool continuation-id descriptor payload]
  (let [resident-route (route pool continuation-id)
        start (long (:chunk/start descriptor))
        token-count (long (:chunk/token-count descriptor))
        end (+ start token-count)
        source-layout (:chunk/layout descriptor)
        source-attention-layout (:attention-state source-layout)
        source-dtype (or (:dtype source-attention-layout) (:dtype source-layout))
        plan (attention-state/payload-plan (:layout pool) token-count)
        expected-elements (reduce + 0 (map :elements plan))]
    (when-not resident-route
      (throw (ex-info "Cannot restore into a nonresident continuation"
                      {:continuation-id continuation-id})))
    (when (:pending resident-route)
      (throw (ex-info "Cannot restore while an append is pending"
                      {:continuation-id continuation-id})))
    (when (or (neg? start) (not (pos? token-count))
              (> end (:token-count resident-route)))
      (throw (ex-info "Chunk range is outside the resident continuation"
                      {:start start :token-count token-count
                       :resident-tokens (:token-count resident-route)})))
    (when-not (= (select-keys source-attention-layout [:kind :token-axis :slabs])
                 (select-keys (:layout pool) [:kind :token-axis :slabs]))
      (throw (ex-info "Chunk attention-state layout differs from the device pool"
                      {:source source-attention-layout :pool (:layout pool)})))
    (when-not (= :half (:dtype pool))
      (throw (ex-info "Chunk restore conversion currently targets FP16 device pools"
                      {:pool-dtype (:dtype pool)})))
    (let [source (half-payload payload source-dtype expected-elements)
          slab-by-name (into {} (map (juxt :name identity)) (:slabs (:layout pool)))
          runs (transfer-runs resident-route start token-count (:page-size pool))
          entries
          (vec
           (mapcat
            (fn [{:keys [slab layer element-offset]}]
              (let [slab-layout (get slab-by-name slab)
                    per-token (:elements-per-token slab-layout)]
                (mapv
                 (fn [{:keys [relative-token physical-token token-count]}]
                   [(token-range-view pool slab-layout layer physical-token token-count)
                    source
                    {:src-element (+ element-offset (* relative-token per-token))
                     :dst-element 0
                     :elements (* token-count per-token)}])
                 runs)))
            plan))]
      (transfer-ranges-measured! pool :upload entries)
      resident-route)))

(defn export-chunk
  "Gather one immutable route range into the durable FP16 chunk format.

  The route is leased for the complete device-to-host transfer, so eviction,
  release, and copy-on-write cannot recycle its pages while capture is in
  flight. Physical page boundaries are independent of the requested durable
  chunk. Current exact-prefix identity assumes absolute positions start at zero;
  compacted nonzero-position routes are rejected until position semantics are
  included in the content identity."
  [pool continuation-id model-fingerprint descriptor]
  (when-not (= :half (:dtype pool))
    (throw (ex-info "Paged chunk export currently requires an FP16 page pool"
                    {:pool-dtype (:dtype pool)})))
  (let [lease (acquire-lease! pool [continuation-id])]
    (try
      (let [resident-route (first (:routes lease))
            start (long (:chunk/start descriptor))
            token-count (long (:chunk/token-count descriptor))
            end (+ start token-count)
            _ (when-not (zero? (long (:start-position resident-route)))
                (throw (ex-info "Durable prefix export requires a zero absolute start position"
                                {:continuation-id continuation-id
                                 :start-position (:start-position resident-route)})))
            _ (when (or (neg? start) (not (pos? token-count))
                        (> end (:token-count resident-route)))
                (throw (ex-info "Chunk range is outside the resident continuation"
                                {:start start :token-count token-count
                                 :resident-tokens (:token-count resident-route)})))
            plan (attention-state/payload-plan (:layout pool) token-count)
            payload-elements (reduce + 0 (map :elements plan))
            halfs (short-array payload-elements)
            slab-by-name (into {} (map (juxt :name identity))
                               (:slabs (:layout pool)))
            runs (transfer-runs resident-route start token-count (:page-size pool))
            entries
            (vec
             (mapcat
              (fn [{:keys [slab layer element-offset]}]
                (let [slab-layout (get slab-by-name slab)
                      per-token (:elements-per-token slab-layout)]
                  (mapv
                   (fn [{:keys [relative-token physical-token token-count]}]
                     [(token-range-view pool slab-layout layer physical-token token-count)
                      halfs
                      {:src-element 0
                       :dst-element (+ element-offset (* relative-token per-token))
                       :elements (* token-count per-token)}])
                   runs)))
              plan))]
        (transfer-ranges-measured! pool :download entries)
        (let [durable-layout (assoc (:layout pool)
                                    :dtype :float16
                                    :byte-order :little-endian)]
          (cond->
           (merge descriptor
                  {:chunk/version 3
                   :chunk/model-fingerprint model-fingerprint
                   :chunk/layout {:dtype :float16
                                  :byte-order :little-endian
                                  :attention-state durable-layout}
                   :chunk/slabs plan
                   :chunk/payload halfs})
            (apply = (map :elements plan))
            (assoc :chunk/elements-per-slab (:elements (first plan))))))
      (finally
        (release-lease! pool lease)))))

(defn append-token!
  "Append one complete host-visible attention-state token to a resident route.

  `values` maps every `[slab-name layer]` to exactly one token row. FP32 rows
  are converted to the pool's FP16 representation. The page reservation is
  committed only after Raster validates and uploads the complete batch; failure
  reverts allocation and copy-on-write metadata before rethrowing.

  This is a correctness and ingestion path. Model inference should eventually
  bind projected resident K/V through Raster's append graph and use
  `reserve-append!`/`commit-append!` around that submission."
  ([pool continuation-id values]
   (append-token! pool continuation-id values {}))
  ([pool continuation-id values {:keys [source-dtype]
                                 :or {source-dtype :float32}}]
   (when-not (= :half (:dtype pool))
     (throw (ex-info "Host token append currently targets FP16 page pools"
                     {:pool-dtype (:dtype pool)})))
   (when-not (= (set (keys (:buffer-keys pool))) (set (keys values)))
     (throw (ex-info "Append values differ from the page pool's slabs and layers"
                     {:expected (set (keys (:buffer-keys pool)))
                      :actual (set (keys values))})))
   (let [reservation (reserve-append! pool continuation-id)]
     (try
       (let [slab-by-name (into {} (map (juxt :name identity))
                                (:slabs (:layout pool)))
             entries
             (mapv (fn [[[slab-name layer] key]]
                     (let [slab-layout (get slab-by-name slab-name)
                           elements (:elements-per-token slab-layout)
                           source (if (= :half (:dtype pool))
                                    (half-payload (get values [slab-name layer])
                                                  source-dtype elements)
                                    (get values [slab-name layer]))]
                       [key source
                        {:dst-element
                         (* (+ (* (:physical-page reservation) (:page-size pool))
                               (:page-offset reservation))
                            elements)
                         :elements elements}]))
                   (:buffer-keys pool))]
         (gpu/upload-ranges! (:session pool) entries)
         (commit-append! pool continuation-id reservation))
       (catch Throwable error
         (abort-append! pool continuation-id reservation)
         (throw error))))))

(defn- dense-values
  [routes pages-per-sequence]
  (let [capacity (long (or pages-per-sequence
                           (max 1 (reduce max 0 (map (comp count :pages) routes)))))]
     (when (or (> capacity Integer/MAX_VALUE)
               (some #(or (> (:token-count %) Integer/MAX_VALUE)
                          (> (:start-position %) Integer/MAX_VALUE))
                     routes))
       (throw (ex-info "Dense route values exceed Raster's int32 coordinates"
                       {:pages-per-sequence capacity
                        :routes routes})))
     (when (some #(> (count (:pages %)) capacity) routes)
       (throw (ex-info "Resident route exceeds dense page-table capacity"
                       {:pages-per-sequence capacity
                        :route-pages (mapv (comp count :pages) routes)})))
     {:page-table
      (int-array
       (mapcat (fn [resident-route]
                 (concat (:pages resident-route)
                         (repeat (- capacity (count (:pages resident-route))) -1)))
               routes))
      :lengths (int-array (map :token-count routes))
      :start-positions (int-array (map :start-position routes))
      :pages-per-sequence capacity}))

(defn dense-route-values
  "Materialize fixed-width dense page-routing arrays for `continuation-ids`.

  This is a point-in-time view for inspection and synchronous setup. Scheduled
  GPU work should use `acquire-lease!` and `leased-dense-route-values` so pages
  cannot be recycled before completion."
  ([pool continuation-ids]
   (dense-route-values pool continuation-ids {}))
  ([pool continuation-ids {:keys [pages-per-sequence]}]
   (let [routes (mapv #(or (route pool %)
                           (throw (ex-info "Batch continuation is not resident"
                                           {:continuation-id %})))
                      continuation-ids)]
     (dense-values routes pages-per-sequence))))

(defn leased-dense-route-values
  "Materialize dense page-routing arrays from a pinned immutable lease snapshot."
  ([pool lease]
   (leased-dense-route-values pool lease {}))
  ([pool lease {:keys [pages-per-sequence]}]
   (when-not (and (page-lease? lease) (identical? pool (:pool lease)))
     (throw (ex-info "Page lease belongs to a different pool"
                     {:lease lease})))
   (when-not (get-in @(:state pool) [:leases (:id lease)])
     (throw (ex-info "Page lease has already been released"
                     {:lease-id (:id lease)})))
   (dense-values (:routes lease) pages-per-sequence)))
