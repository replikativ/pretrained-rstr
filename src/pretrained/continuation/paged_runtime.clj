(ns pretrained.continuation.paged-runtime
  "Device-local continuous batching for paged continuation workers.

  One runtime thread owns all decoder calls. Controller handler threads enqueue
  restore operations, incremental prompt-prefill jobs, and decode jobs, then
  wait on fenced futures. Each graph iteration uses the pure scheduler and the
  paged decoder's sparse fixed-lane API; no tensor data enters the controller
  or Kabel protocols."
  (:require [pretrained.continuation.page-pool :as page-pool]
            [pretrained.continuation.paged-decoder :as paged-decoder]
            [pretrained.continuation.scheduler :as scheduler])
  (:import (java.io Closeable)
           (java.util UUID)
           (java.util.concurrent CancellationException CompletableFuture
                                 ExecutionException ExecutorService Executors
                                 Future LinkedBlockingQueue TimeUnit)))

(def ^:private lane-job-types #{:prefill :decode})

(defn- continuation-id
  [effect]
  (let [request (:assignment/request effect)]
    (or (:request/continuation-id request) (:request/id request))))

(defn- complete!
  [runtime job value]
  (swap! (:registry runtime) dissoc (:job/id job))
  (.complete ^CompletableFuture (:job/future job) value))

(defn- fail!
  [runtime job error]
  (swap! (:registry runtime) dissoc (:job/id job))
  (.completeExceptionally ^CompletableFuture (:job/future job) error))

(defn- cancelled?
  [job]
  (or @(:job/cancelled? job)
      (.isCancelled ^CompletableFuture (:job/future job))))

(defn- discard-cancelled
  [runtime jobs]
  (reduce (fn [retained job]
            (if (cancelled? job)
              (do
                (swap! (:registry runtime) dissoc (:job/id job))
                (.cancel ^CompletableFuture (:job/future job) false)
                retained)
              (conj retained job)))
          [] jobs))

(defn- drain-inbound
  [^LinkedBlockingQueue inbound]
  (loop [items []]
    (if-let [item (.poll inbound)]
      (recur (conj items item))
      items)))

(defn- execute-operation!
  [runtime job]
  (if (cancelled? job)
    (fail! runtime job (CancellationException. "Paged operation was cancelled"))
    (try
      (let [result ((:job/operation job))]
        (if (cancelled? job)
          (fail! runtime job
                 (CancellationException. "Paged operation was cancelled"))
          (complete! runtime job result)))
      (catch Throwable error
        (fail! runtime job error)))))

(defn- request-jobs
  [lanes deferred]
  (into (filterv some? lanes) deferred))

(defn- lane-summary
  [lanes]
  (mapv #(some-> %
                 (select-keys
                  [:request/id :request/phase :request/continuation-id
                   :request/remaining-tokens]))
        lanes))

(defn- bulk-prefill-job
  [runtime lane-plan]
  (let [item (first (:lanes lane-plan))
        tile-size (:prefill-tile-size runtime)]
    (when (and (= 1 (:capacity runtime))
               (:prefill-range! runtime)
               item
               (= :prefill (:request/phase item))
               (<= tile-size (:request/remaining-tokens item)))
      item)))

(defn- execute-bulk-prefill!
  [runtime lane-plan deferred job]
  (let [tile-size (:prefill-tile-size runtime)
        position (long (:request/position job))
        tokens (subvec (:job/prompt job) position (+ position tile-size))
        started (System/nanoTime)]
    ((:prefill-range! runtime) (:decoder runtime)
     (:request/continuation-id job) tokens position)
    (let [elapsed-nanos (- (System/nanoTime) started)
          cancelled (cancelled? job)
          remaining (- (long (:request/remaining-tokens job)) tile-size)
          next-job (when (and (not cancelled) (pos? remaining))
                     (-> job
                         (dissoc :scheduled/tokens)
                         (assoc :request/position (+ position tile-size)
                                :request/remaining-tokens remaining)))
          next-lanes [next-job]]
      (swap! (:state runtime)
             (fn [state]
               (-> state
                   (update :iterations inc)
                   (update :scheduled-tokens + tile-size)
                   (update :bulk-prefill-tiles (fnil inc 0))
                   (update :bulk-prefill-tokens (fnil + 0) tile-size)
                   (update :bulk-prefill-nanos (fnil + 0) elapsed-nanos)
                   (assoc :lanes (lane-summary next-lanes)
                          :deferred-count (count deferred)))))
      (cond
        cancelled
        (fail! runtime job
               (CancellationException. "Paged lane was cancelled"))

        (zero? remaining)
        (complete! runtime job {:ok? true})

        :else nil)
      {:lanes next-lanes :deferred deferred})))

(defn- prime-values
  [lane-plan]
  (let [refill-ids (into #{} (map :request/id) (:refill lane-plan))]
    (into []
          (keep (fn [{:keys [lane/index]
                      :request/keys [id phase position pending-token]
                      :job/keys [prompt]}]
                  (cond
                    (= :prefill phase)
                    {:lane index :token (nth prompt position)}

                    (contains? refill-ids id)
                    {:lane index :token pending-token}

                    :else nil)))
          (:lanes lane-plan))))

(defn- lane-work
  [lanes]
  (into []
        (keep (fn [{:keys [lane/index]
                    :request/keys [continuation-id position]}]
                (when (some? index)
                  {:lane index
                   :continuation-id continuation-id
                   :position position})))
        lanes))

(defn- stop-reason
  [runtime job token remaining]
  (cond
    (zero? remaining) :length
    (contains? (:eos-ids runtime) token) :eos
    (>= (long ((:route-token-count runtime)
               (:decoder runtime) (:request/continuation-id job)))
        (:max-position runtime)) :capacity
    :else nil))

(defn- advance-lane
  [runtime result job]
  (if (cancelled? job)
    {:next-lane nil
     :completion [:fail
                  (CancellationException. "Paged lane was cancelled")]}
    (let [position (inc (long (:request/position job)))
          remaining (dec (long (:request/remaining-tokens job)))
          job (-> job
                  (dissoc :scheduled/tokens)
                  (assoc :request/position position
                         :request/remaining-tokens remaining))]
      (case (:request/phase job)
        :prefill
        (if (zero? remaining)
          {:next-lane nil :completion [:complete {:ok? true}]}
          {:next-lane job})

        :decode
        (let [token (long (:token result))
              output (conj (:job/output job) token)
              reason (stop-reason runtime job token remaining)]
          (when-let [token! (:job/token! job)]
            ;; The callback must enqueue consumer delivery without blocking the
            ;; device-owning loop. Assignment fencing happens in the worker and
            ;; router state machines, not in this callback.
            (token! token (dec (count output))))
          (if reason
            {:next-lane nil
             :completion [:complete
                          {:ok? true :tokens output :stop-reason reason}]}
            {:next-lane
             (assoc job
                    :request/pending-token token
                    :job/output output)}))

        (throw (ex-info "Paged runtime encountered an unsupported lane phase"
                        {:phase (:request/phase job)}))))))

(defn- execute-lanes!
  [runtime lanes deferred]
  (let [jobs (discard-cancelled runtime (request-jobs lanes deferred))]
    (if (empty? jobs)
      (let [lanes (vec (repeat (:capacity runtime) nil))]
        (swap! (:state runtime) assoc :lanes lanes :deferred-count 0)
        {:lanes lanes :deferred []})
      (let [iteration
            (scheduler/plan-iteration
             {:max-batched-tokens (:capacity runtime)
              :max-sequences (:capacity runtime)
              :prefill-chunk-size 1
              :repair-chunk-size 1
              :minimum-prefill-tokens (:minimum-prefill-tokens runtime)}
             jobs)
            lane-plan (scheduler/plan-work-lanes
                       (:capacity runtime) lanes (:scheduled iteration))
            deferred (into (vec (:deferred iteration))
                           (:deferred lane-plan))]
        (if-let [bulk-job (bulk-prefill-job runtime lane-plan)]
          (execute-bulk-prefill! runtime lane-plan deferred bulk-job)
          (let [primes (prime-values lane-plan)
                work (lane-work (:lanes lane-plan))]
            (when (seq primes)
              ((:prime-lanes! runtime) (:decoder runtime) primes))
            (let [results ((:step-lanes! runtime) (:decoder runtime) work)
                  by-lane (into {} (map (juxt :lane identity)) results)
                  advanced
                  (mapv (fn [{:keys [lane/index] :as job}]
                          (when job
                            (assoc (advance-lane runtime (get by-lane index) job)
                                   :job job)))
                        (:lanes lane-plan))
                  next-lanes (mapv :next-lane advanced)]
              (swap! (:state runtime)
                     (fn [state]
                       (-> state
                           (update :iterations inc)
                           (update :scheduled-tokens + (count work))
                           (assoc :lanes (lane-summary next-lanes)
                                  :deferred-count (count deferred)))))
              (doseq [{:keys [job completion]} advanced
                      :when completion]
                (let [[action value] completion]
                  (case action
                    :complete (complete! runtime job value)
                    :fail (fail! runtime job value))))
              {:lanes next-lanes :deferred deferred})))))))

(defn- fail-runtime!
  [runtime error]
  (reset! (:closed? runtime) true)
  (doseq [[_ job] @(:registry runtime)]
    (reset! (:job/cancelled? job) true)
    (fail! runtime job error))
  ((:on-error! runtime) error))

(defn- run-loop!
  [runtime]
  (let [inbound ^LinkedBlockingQueue (:inbound runtime)]
    (try
      (loop [operations []
             deferred []
             lanes (vec (repeat (:capacity runtime) nil))]
        (when-not @(:closed? runtime)
          (let [idle? (and (empty? operations)
                           (empty? deferred)
                           (every? nil? lanes))
                first-item (when idle? (.take inbound))
                incoming (cond-> []
                           first-item (conj first-item)
                           true (into (drain-inbound inbound)))
                operations (into operations
                                 (filter #(= :operation (:job/type %))) incoming)
                new-jobs (filter #(contains? lane-job-types (:job/type %)) incoming)
                deferred (into deferred new-jobs)
                operation (first operations)
                operations (if operation (subvec (vec operations) 1) operations)
                _ (when operation (execute-operation! runtime operation))
                {next-lanes :lanes next-deferred :deferred}
                (execute-lanes! runtime lanes deferred)]
            (recur operations next-deferred next-lanes))))
      (catch InterruptedException _
        nil)
      (catch Throwable error
        (when-not @(:closed? runtime)
          (fail-runtime! runtime error))))))

(defrecord PagedRuntime
    [decoder capacity minimum-prefill-tokens eos-ids max-position prime-lanes!
     step-lanes! prefill-range! prefill-tile-size route-token-count inbound
     registry arrival state thread on-error! closed?]
  Closeable
  (close [this]
    (locking this
      (when (compare-and-set! closed? false true)
        (.interrupt ^Thread thread)
        (let [error (ex-info "Paged runtime is closed" {})]
          (doseq [[_ job] @registry]
            (reset! (:job/cancelled? job) true)
            (fail! this job error)))))))

(defn open-runtime
  "Open one continuous-batching loop for a fixed-batch paged decoder.

  Options may override `:capacity`, `:minimum-prefill-tokens`, `:eos-ids`, and
  the decoder operations for simulation. Capacity defaults to the decoder's
  bound batch size. A decoder bound with `:prefill-T` uses one exact multi-row
  tile per single-lane scheduler turn; decode work can run before the next tile.
  Short prefill tails retain the ordinary lane path. A multi-lane runtime
  reserves one lane-token for waiting prefill work by default; a single lane
  finishes active decode before admitting another prompt. Close the runtime
  after its worker controller has quiesced."
  ([decoder] (open-runtime decoder {}))
  ([decoder {:keys [capacity minimum-prefill-tokens eos-ids max-position
                    prime-lanes! step-lanes! prefill-range! prefill-tile-size
                    route-token-count on-error!]
             :or {eos-ids #{}
                  on-error! (fn [error] (.printStackTrace ^Throwable error))}}]
   (let [capacity (or capacity (get-in decoder [:decode-state :batch-size]) 1)
         decoder-capacity (get-in decoder [:decode-state :batch-size])
         minimum-prefill-tokens
         (if (nil? minimum-prefill-tokens)
           (if (> (long capacity) 1) 1 0)
           minimum-prefill-tokens)
         max-position (or max-position (get-in decoder [:decode-state :maxpos]))
         prefill-tile-size (or prefill-tile-size (:prefill-T decoder))
         prefill-range! (or prefill-range!
                            (when (and prefill-tile-size
                                       (:prefill-executable decoder))
                              paged-decoder/prefill-range!))]
     (when-not (and (integer? capacity) (pos? capacity))
       (throw (ex-info "Paged runtime capacity must be a positive integer"
                       {:capacity capacity})))
     (when (and (integer? decoder-capacity) (> capacity decoder-capacity))
       (throw (ex-info "Paged runtime exceeds the decoder's fixed batch"
                       {:capacity capacity
                        :decoder-capacity decoder-capacity})))
     (when-not (and (integer? minimum-prefill-tokens)
                    (<= 0 minimum-prefill-tokens capacity))
       (throw (ex-info "Paged runtime prefill reservation is invalid"
                       {:minimum-prefill-tokens minimum-prefill-tokens
                        :capacity capacity})))
     (when-not (and (integer? max-position) (pos? max-position))
       (throw (ex-info "Paged runtime requires a positive maximum position"
                       {:max-position max-position})))
     (when (and prefill-range!
                (not (and (= 1 (long capacity))
                          (integer? prefill-tile-size)
                          (pos? prefill-tile-size))))
       (throw (ex-info "Bulk prefill requires a positive tile and one runtime lane"
                       {:capacity capacity
                        :prefill-tile-size prefill-tile-size})))
     (when (and prefill-range! (:prefill-T decoder)
                (not= (long prefill-tile-size) (long (:prefill-T decoder))))
       (throw (ex-info "Runtime tile must match the decoder's compiled prefill rows"
                       {:runtime-prefill-tile-size prefill-tile-size
                        :decoder-prefill-T (:prefill-T decoder)})))
     (when-not (or (nil? prefill-range!) (ifn? prefill-range!))
       (throw (ex-info "Paged runtime bulk-prefill callback must be callable"
                       {:prefill-range! prefill-range!})))
     (when-not (every? ifn?
                       [(or prime-lanes! paged-decoder/prime-lanes!)
                        (or step-lanes! paged-decoder/step-lanes!)
                        (or route-token-count (constantly 0))
                        on-error!])
       (throw (ex-info "Paged runtime callbacks must be callable" {})))
     (when-not (every? integer? eos-ids)
       (throw (ex-info "Paged runtime EOS ids must be integers"
                       {:eos-ids eos-ids})))
     (let [holder (atom nil)
           runtime (map->PagedRuntime
                    {:decoder decoder
                     :capacity (long capacity)
                     :minimum-prefill-tokens (long minimum-prefill-tokens)
                     :eos-ids (set eos-ids)
                     :max-position (long max-position)
                     :prime-lanes! (or prime-lanes! paged-decoder/prime-lanes!)
                     :step-lanes! (or step-lanes! paged-decoder/step-lanes!)
                     :prefill-range! prefill-range!
                     :prefill-tile-size (some-> prefill-tile-size long)
                     :route-token-count
                     (or route-token-count
                         (fn [decoder continuation-id]
                           (:token-count
                            (page-pool/route (:pool decoder) continuation-id))))
                     :inbound (LinkedBlockingQueue.)
                     :registry (atom {})
                     :arrival (atom -1)
                     :state (atom {:iterations 0 :scheduled-tokens 0
                                   :bulk-prefill-tiles 0
                                   :bulk-prefill-tokens 0
                                   :bulk-prefill-nanos 0
                                   :lanes (vec (repeat capacity nil))
                                   :deferred-count 0})
                     :on-error! on-error!
                     :closed? (atom false)})
           thread (Thread. ^Runnable
                           (reify Runnable
                             (run [_] (run-loop! @holder)))
                           "pretrained-paged-runtime")
           runtime (assoc runtime :thread thread)]
       (.setDaemon thread true)
       (reset! holder runtime)
       (.start thread)
       runtime))))

(defn state
  "Return immutable counters, tile timing, lane summaries, and queue depth."
  [runtime]
  (let [snapshot @(:state runtime)
        tokens (long (:bulk-prefill-tokens snapshot 0))
        milliseconds (/ (long (:bulk-prefill-nanos snapshot 0)) 1.0e6)]
    (cond->
     (assoc snapshot
            :prefill-tile-size (when (:prefill-range! runtime)
                                 (:prefill-tile-size runtime))
            :bulk-prefill-milliseconds milliseconds
            :active-job-count (count @(:registry runtime))
            :inbound-queue-depth (.size ^LinkedBlockingQueue (:inbound runtime)))
      (pos? tokens) (assoc :bulk-prefill-ms-per-token (/ milliseconds tokens)))))

(defn- await-job!
  [runtime job]
  (locking runtime
    (when @(:closed? runtime)
      (throw (ex-info "Paged runtime is closed" {})))
    (swap! (:registry runtime) assoc (:job/id job) job)
    (.put ^LinkedBlockingQueue (:inbound runtime) job))
  (try
    (.get ^CompletableFuture (:job/future job))
    (catch InterruptedException error
      (reset! (:job/cancelled? job) true)
      (.offer ^LinkedBlockingQueue (:inbound runtime) {:job/type :wake})
      (.interrupt (Thread/currentThread))
      (throw (ex-info "Paged controller operation was cancelled"
                      {:job/id (:job/id job)} error)))
    (catch ExecutionException error
      (throw (.getCause error)))))

(defn- job
  [runtime type values]
  (merge values
         {:job/id (UUID/randomUUID)
          :job/type type
          :job/future (CompletableFuture.)
          :job/cancelled? (atom false)
          :request/arrival (swap! (:arrival runtime) inc)}))

(defn run-operation!
  "Run one synchronous device/storage operation between decoder iterations.

  The optional `assignment-id` arity lets controller cancellation fence a
  queued or in-flight operation at its completion boundary."
  ([runtime operation]
   (run-operation! runtime nil operation))
  ([runtime assignment-id operation]
   (when-not (ifn? operation)
     (throw (ex-info "Paged runtime operation must be callable" {})))
   (await-job! runtime
               (job runtime :operation
                    {:job/assignment-id assignment-id
                     :job/operation operation}))))

(defn run-background-operation!
  "Run storage/transfer work on the calling handler thread with runtime fencing.

  `operation` receives a zero-argument cancellation predicate. The job is
  registered by assignment so controller cancellation becomes visible without
  interrupting an active storage or GPU transfer. The operation must stop
  admitting new work and return or throw only after its current safe boundary."
  [runtime assignment-id operation]
  (when-not (ifn? operation)
    (throw (ex-info "Paged background operation must be callable" {})))
  (let [job (job runtime :background {:job/assignment-id assignment-id})]
    (locking runtime
      (when @(:closed? runtime)
        (throw (ex-info "Paged runtime is closed" {})))
      (swap! (:registry runtime) assoc (:job/id job) job))
    (try
      (let [result (operation #(cancelled? job))]
        (locking runtime
          (if (cancelled? job)
            (let [error (CancellationException.
                         "Paged background operation was cancelled")]
              (fail! runtime job error)
              (throw error))
            (do
              (complete! runtime job result)
              result))))
      (catch Throwable error
        (when (contains? @(:registry runtime) (:job/id job))
          (fail! runtime job error))
        (throw error)))))

(defn prefill!
  "Incrementally compute an assignment's uncached prompt suffix in batch lanes."
  [runtime effect]
  (let [request (:assignment/request effect)
        prompt (vec (:request/tokens request))
        id (continuation-id effect)]
    (when-not (seq prompt)
      (throw (ex-info "Paged runtime prefill requires a nonempty prompt" {})))
    (when-not (every? integer? prompt)
      (throw (ex-info "Paged runtime prompt tokens must be integers" {})))
    (let [cached (long ((:route-token-count runtime) (:decoder runtime) id))
          processed (dec (count prompt))
          remaining (- processed cached)]
      (when (neg? remaining)
        (throw (ex-info "Resident prefix is longer than the submitted prompt"
                        {:continuation-id id :cached cached
                         :processed processed})))
      (if (zero? remaining)
        {:ok? true}
        (await-job!
         runtime
         (job runtime :prefill
              {:job/assignment-id (:assignment/id effect)
               :request/id [(:assignment/id effect) :prefill]
               :request/phase :prefill
               :request/remaining-tokens remaining
               :request/continuation-id id
               :request/position cached
               :job/prompt prompt}))))))

(defn decode!
  "Generate one assignment in shared decode lanes until EOS or its token budget."
  [runtime effect]
  (let [request (:assignment/request effect)
        prompt (vec (:request/tokens request))
        id (continuation-id effect)]
    (when-not (seq prompt)
      (throw (ex-info "Paged runtime decode requires a nonempty prompt" {})))
    (when-not (every? integer? prompt)
      (throw (ex-info "Paged runtime prompt tokens must be integers" {})))
    (when-not (and (integer? (:request/max-new-tokens request))
                   (pos? (:request/max-new-tokens request)))
      (throw (ex-info "Paged runtime decode requires a positive token budget"
                      {:max-new-tokens (:request/max-new-tokens request)})))
    (let [processed (dec (count prompt))
          resident (long ((:route-token-count runtime) (:decoder runtime) id))]
      (when-not (= processed resident)
        (throw (ex-info "Paged decode requires a completely primed prompt"
                        {:continuation-id id
                         :processed-token-count processed
                         :resident-token-count resident})))
      (await-job!
       runtime
       (job runtime :decode
            {:job/assignment-id (:assignment/id effect)
             :request/id [(:assignment/id effect) :decode]
             :request/phase :decode
             :request/remaining-tokens (:request/max-new-tokens request)
             :request/continuation-id id
             :request/position processed
             :request/pending-token (peek prompt)
             :job/token! (:worker/token! effect)
             :job/output []})))))

(defn cancel-assignment!
  "Mark every queued or active job for `assignment-id` as cancelled.

  The device loop observes the mark at its next atomic operation/graph boundary
  and only then wakes waiting controller tasks, so capacity is not released
  underneath an in-flight append. Returns the number of marked jobs."
  [runtime assignment-id]
  (locking runtime
    (let [jobs (filter #(= assignment-id (:job/assignment-id %))
                       (vals @(:registry runtime)))]
      (doseq [job jobs]
        (reset! (:job/cancelled? job) true))
      (when (seq jobs)
        (.offer ^LinkedBlockingQueue (:inbound runtime) {:job/type :wake}))
      (count jobs))))

(defn concurrent-submission
  "Return bounded concurrent submission callbacks for a local controller.

  Handler tasks may block while the device loop batches their jobs, so the
  controller's default single thread is insufficient. Future cancellation does
  not interrupt running handlers; use `controller-submission` to additionally
  cancel their runtime jobs. `max-concurrency` defaults to 64 assignments."
  ([] (concurrent-submission 64))
  ([max-concurrency]
   (when-not (and (integer? max-concurrency) (pos? max-concurrency))
     (throw (ex-info "Controller concurrency must be a positive integer"
                     {:max-concurrency max-concurrency})))
   (let [executor (Executors/newFixedThreadPool (int max-concurrency))]
     {:submit! (fn [task]
                 (.submit ^ExecutorService executor
                          ^Runnable (reify Runnable (run [_] (task)))))
      :cancel! (fn [task]
                 (when task (.cancel ^Future task false)))
      :close-submit! (fn []
                       (.shutdown ^ExecutorService executor)
                       (when-not (.awaitTermination ^ExecutorService executor
                                                    30 TimeUnit/SECONDS)
                         (.shutdownNow ^ExecutorService executor)))})))

(defn controller-submission
  "Return concurrent local-controller callbacks with safe runtime cancellation."
  ([runtime] (controller-submission runtime 64))
  ([runtime max-concurrency]
   (assoc (concurrent-submission max-concurrency)
          :cancel-operation!
          #(cancel-assignment! runtime (:assignment/id %)))))
