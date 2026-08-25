(ns pretrained.continuation.paged-append
  "Batched resident K/V assignment into continuation page pools.

  Page allocation is separated from per-layer GPU writes. One reservation batch
  may therefore be loaded into every transformer layer's append runner and is
  committed only after all layer events complete. Raster owns dtype conversion,
  device assignment, and queue ordering; the page pool owns copy-on-write and
  publication of the new logical length."
  (:refer-clojure :exclude [run!])
  (:require [pretrained.continuation.page-pool :as page-pool]
            [raster.compiler.ir.paged-kv-append :as raster-append]
            [raster.compiler.passes.parallel.paged-kv-append-route :as append-route]
            [raster.gpu.core :as gpu])
  (:import [java.io Closeable]
           [java.util UUID]))

(declare close-runner!)

(defrecord AppendBatch [pool id entries slots state])

(defrecord PagedAppendRunner
           [session pool layer problem handle slot-key graph-key state]
  Closeable
  (close [runner]
    (close-runner! runner)))

(defn append-batch?
  "Return true when `value` is a paged append reservation batch."
  [value]
  (instance? AppendBatch value))

(defn runner?
  "Return true when `value` is a paged append runner."
  [value]
  (instance? PagedAppendRunner value))

(defn- checked-positive
  [field value]
  (when-not (and (integer? value) (pos? value))
    (throw (ex-info "Paged append capacity must be a positive integer"
                    {:field field :value value})))
  (long value))

(defn- physical-slot
  [pool reservation]
  (let [slot (try
               (Math/addExact
                (Math/multiplyExact (long (:physical-page reservation))
                                    (long (:page-size pool)))
                (long (:page-offset reservation)))
               (catch ArithmeticException error
                 (throw (ex-info "Paged append slot coordinate overflow"
                                 {:reservation reservation} error))))]
    (when (> slot Integer/MAX_VALUE)
      (throw (ex-info "Paged append slot exceeds Raster's int32 descriptor"
                      {:slot slot :reservation reservation})))
    slot))

(defn reserve-batch!
  "Reserve one append slot per continuation and return an `AppendBatch`.

  Continuation identities must be unique. A partial reservation failure rolls
  back every reservation already obtained. Complete the batch exactly once with
  `commit-batch!` or `abort-batch!`."
  [pool continuation-ids]
  (when-not (page-pool/page-pool? pool)
    (throw (ex-info "Paged append requires a DevicePagePool" {:pool pool})))
  (let [continuation-ids (vec continuation-ids)]
    (when-not (seq continuation-ids)
      (throw (ex-info "Paged append requires at least one continuation" {})))
    (when-not (= (count continuation-ids) (count (set continuation-ids)))
      (throw (ex-info "Paged append continuation identities must be unique"
                      {:continuation-ids continuation-ids})))
    (loop [remaining continuation-ids
           entries []]
      (if-let [continuation-id (first remaining)]
        (let [attempt (try
                        {:reservation
                         (page-pool/reserve-append! pool continuation-id)}
                        (catch Throwable error {:error error}))]
          (if-let [error (:error attempt)]
            (do
              (when (seq entries)
                (page-pool/abort-appends! pool entries))
              (throw error))
            (recur (next remaining)
                   (conj entries
                         {:continuation-id continuation-id
                          :reservation (:reservation attempt)}))))
        (try
          (map->AppendBatch
           {:pool pool
            :id (UUID/randomUUID)
            :entries entries
            :slots (int-array
                    (map #(physical-slot pool (:reservation %)) entries))
            :state (atom :reserved)})
          (catch Throwable error
            (page-pool/abort-appends! pool entries)
            (throw error)))))))

(defn reservation-entries
  "Return the ordered page-pool reservation entries in `batch`."
  [batch]
  (when-not (append-batch? batch)
    (throw (ex-info "Expected an AppendBatch" {:batch batch})))
  (:entries batch))

(defn slot-values
  "Return a copy of the ordered int32 physical slots in `batch`."
  [batch]
  (when-not (append-batch? batch)
    (throw (ex-info "Expected an AppendBatch" {:batch batch})))
  (aclone ^ints (:slots batch)))

(defn commit-batch!
  "Publish a completed append batch and return its updated resident routes."
  [batch]
  (when-not (append-batch? batch)
    (throw (ex-info "Expected an AppendBatch" {:batch batch})))
  (locking batch
    (when-not (= :reserved @(:state batch))
      (throw (ex-info "Append batch has already been completed"
                      {:batch-id (:id batch) :state @(:state batch)})))
    (let [routes (page-pool/commit-appends! (:pool batch) (:entries batch))]
      (reset! (:state batch) :committed)
      routes)))

(defn abort-batch!
  "Revert an uncommitted append batch and return its restored resident routes."
  [batch]
  (when-not (append-batch? batch)
    (throw (ex-info "Expected an AppendBatch" {:batch batch})))
  (locking batch
    (when-not (= :reserved @(:state batch))
      (throw (ex-info "Append batch has already been completed"
                      {:batch-id (:id batch) :state @(:state batch)})))
    (let [routes (page-pool/abort-appends! (:pool batch) (:entries batch))]
      (reset! (:state batch) :aborted)
      routes)))

(defn- slab
  [pool slab-name layer]
  (let [slab-layout (some #(when (= slab-name (:name %)) %)
                          (:slabs (:layout pool)))]
    (when-not slab-layout
      (throw (ex-info "Paged append requires key and value slabs"
                      {:missing slab-name :layout (:layout pool)})))
    (when-not (and (<= 0 layer) (< layer (:count slab-layout)))
      (throw (ex-info "Paged append layer is outside its state slab"
                      {:slab slab-name :layer layer :layers (:count slab-layout)})))
    slab-layout))

(defn reference-plan
  "Build one layer's routed append graph without allocating GPU resources.

  Required options are `:layer`, `:batch-size`, `:key-view`, and `:value-view`.
  Both views are caller-owned contiguous FP32 Raster resident views containing
  one projected row per batch lane. The pool must use FP16 storage."
  [pool {:keys [id layer batch-size key-view value-view key-prefix] :as opts}]
  (when-not (page-pool/page-pool? pool)
    (throw (ex-info "Paged append requires a DevicePagePool" {:pool pool})))
  (when-not (= :half (:dtype pool))
    (throw (ex-info "Raster paged append currently requires an FP16 page pool"
                    {:pool-dtype (:dtype pool)})))
  (doseq [[role view] [[:key key-view] [:value value-view]]]
    (when-not (gpu/resident-buffer-view? view)
      (throw (ex-info "Paged append sources must be Raster resident views"
                      {:role role :view view :actual (type view)}))))
  (let [layer (long (or layer 0))
        batch-size (checked-positive :batch-size batch-size)
        key-slab (slab pool :key layer)
        value-slab (slab pool :value layer)
        prefix (or key-prefix (str "paged-append-" (UUID/randomUUID)))
        ids (into {}
                  (map (fn [role] [role (keyword (str prefix "/" (name role)))]))
                  [:key-rows :value-rows :slots :key-pages :value-pages])
        slot-key (keyword (str prefix "-slots"))
        problem (raster-append/make
                 {:id (or id [:pretrained-paged-append prefix layer])
                  :key-rows (:key-rows ids)
                  :value-rows (:value-rows ids)
                  :slot-mapping (:slots ids)
                  :key-pages (:key-pages ids)
                  :value-pages (:value-pages ids)
                  :batch-size batch-size
                  :key-elements-per-token (:elements-per-token key-slab)
                  :value-elements-per-token (:elements-per-token value-slab)
                  :page-size (:page-size pool)
                  :physical-pages (:physical-pages pool)})
        routed (append-route/route! problem)
        bindings {(:key-rows ids) key-view
                  (:value-rows ids) value-view
                  (:slots ids) slot-key
                  (:key-pages ids) (get (page-pool/buffer-keys pool) [:key layer])
                  (:value-pages ids) (get (page-pool/buffer-keys pool) [:value layer])}]
    {:problem problem
     :graph (:graph routed)
     :strategy (:strategy routed)
     :bindings bindings
     :slot-key slot-key
     :allocation [slot-key [:int batch-size nil :input]]
     :options (assoc opts :layer layer :batch-size batch-size)}))

(defn open-runner!
  "Allocate and bind one fixed-batch, single-layer paged append runner.

  The caller owns the page pool, session, and source views. Closing the runner
  releases only its graph and slot descriptor buffer."
  [pool opts]
  (let [{:keys [problem graph bindings slot-key allocation options]}
        (reference-plan pool opts)
        session (:session pool)
        graph-key (or (:graph-key opts) [:paged-append (:id problem)])]
    (try
      (gpu/alloc! session {(first allocation) (second allocation)})
      (let [handle (gpu/bind-kernel-graph! session graph-key graph bindings {})]
        (map->PagedAppendRunner
         {:session session
          :pool pool
          :layer (:layer options)
          :problem problem
          :handle handle
          :slot-key slot-key
          :graph-key graph-key
          :state (atom {:closed? false :pending nil :batch nil})}))
      (catch Throwable error
        (when (gpu/buffer session slot-key)
          (gpu/free-buffer! session slot-key))
        (throw error)))))

(defn load-batch!
  "Validate and upload one reservation batch's small slot descriptor."
  [runner batch]
  (when-not (runner? runner)
    (throw (ex-info "Expected a PagedAppendRunner" {:runner runner})))
  (when-not (append-batch? batch)
    (throw (ex-info "Expected an AppendBatch" {:batch batch})))
  (locking runner
    (let [{:keys [closed? pending] loaded-batch :batch} @(:state runner)]
      (when closed?
        (throw (ex-info "Paged append runner is closed" {})))
      (when pending
        (throw (ex-info "Cannot replace a paged append batch while in flight"
                        {:event pending})))
      (when (and loaded-batch
                 (not= (:id loaded-batch) (:id batch)))
        (throw (ex-info "Paged append runner already has a loaded batch"
                        {:loaded-batch-id (:id loaded-batch)
                         :batch-id (:id batch)})))
      (when-not (identical? (:pool runner) (:pool batch))
        (throw (ex-info "Append batch belongs to a different page pool"
                        {:batch-id (:id batch)})))
      (when-not (= :reserved @(:state batch))
        (throw (ex-info "Append batch is no longer reserved"
                        {:batch-id (:id batch) :state @(:state batch)})))
      (let [slots (:slots batch)]
        (raster-append/validate-slot-values! (:problem runner) slots)
        (gpu/upload! (:session runner) (:slot-key runner) slots)
        (swap! (:state runner) assoc :batch batch)
        runner))))

(defn submit!
  "Submit the loaded assignment graph without blocking and return its GPU event."
  [runner]
  (locking runner
    (let [{:keys [closed? pending batch]} @(:state runner)]
      (when closed?
        (throw (ex-info "Paged append runner is closed" {})))
      (when pending
        (throw (ex-info "Paged append runner already has an in-flight graph"
                        {:event pending})))
      (when-not batch
        (throw (ex-info "Paged append runner has no loaded reservation batch" {})))
      (when-not (= :reserved @(:state batch))
        (throw (ex-info "Loaded append batch is no longer reserved"
                        {:batch-id (:id batch) :state @(:state batch)})))
      (let [event (gpu/submit-kernel-graph! (:session runner) (:handle runner))]
        (swap! (:state runner) assoc :pending event)
        event))))

(defn await!
  "Wait for one assignment event and return its still-uncommitted append batch."
  [runner event]
  (locking runner
    (let [{:keys [pending batch]} @(:state runner)]
      (when-not (= event pending)
        (throw (ex-info "GPU event does not belong to this append submission"
                        {:expected pending :actual event})))
      (try
        (gpu/await-event! (:session runner) event)
        batch
        (finally
          (gpu/release-event! (:session runner) event)
          (swap! (:state runner) assoc :pending nil :batch nil))))))

(defn run!
  "Load and synchronously execute one layer write, returning the append batch."
  [runner batch]
  (load-batch! runner batch)
  (await! runner (submit! runner)))

(defn close-runner!
  "Wait for pending work and release runner-owned graph resources. Idempotent."
  [runner]
  (locking runner
    (when-not (:closed? @(:state runner))
      (when-let [event (:pending @(:state runner))]
        (gpu/release-event! (:session runner) event))
      (gpu/release-kernel-graph! (:session runner) (:handle runner))
      (when (gpu/buffer (:session runner) (:slot-key runner))
        (gpu/free-buffer! (:session runner) (:slot-key runner)))
      (swap! (:state runner) assoc :closed? true :pending nil :batch nil)))
  nil)
