(ns pretrained.continuation.controller.sim
  "Deterministic logical-time simulator for continuation placement and workers.

  A simulation is an immutable value. The simulator interprets the same router
  and worker effects intended for Kabel, storage, and Raster adapters. Directed
  links are FIFO for equal delivery times; explicit partitions, pauses, crashes,
  and restarts are values rather than wall-clock races. Numerical kernels are
  represented only by deterministic logical costs and token results."
  (:require [pretrained.continuation.controller.router :as router]
            [pretrained.continuation.controller.worker :as worker]))

(defn make-sim
  "Create a virtual continuation cluster.

  `workers` maps worker ids to `worker/initial-state` option maps or initialized
  worker states. Options accept non-negative `:network-delay` (default 1) and
  positive `:offer-timeout` (default 10), both in logical ticks."
  ([workers] (make-sim workers {}))
  ([workers {:keys [network-delay offer-timeout]
             :or {network-delay 1 offer-timeout 10}}]
   (when-not (and (integer? network-delay) (not (neg? network-delay)))
     (throw (ex-info "Simulation network delay must be a non-negative integer"
                     {:network-delay network-delay})))
   (when-not (and (integer? offer-timeout) (pos? offer-timeout))
     (throw (ex-info "Simulation offer timeout must be a positive integer"
                     {:offer-timeout offer-timeout})))
   {:sim/time 0
    :sim/sequence 0
    :sim/router (router/initial-state)
    :sim/workers
    (into (sorted-map)
          (map (fn [[worker-id value]]
                 [worker-id
                  (if (:worker/assignments value)
                    value
                    (worker/initial-state (assoc value :worker/id worker-id)))]))
          workers)
    :sim/events []
    :sim/responses []
    :sim/trace []
    :sim/paused #{}
    :sim/blocked-links #{}
    :sim/config {:network-delay (long network-delay)
                 :offer-timeout (long offer-timeout)}}))

(defn- tick-delay
  [value]
  (long (Math/ceil (double (max 0 value)))))

(defn- schedule
  [sim delay target event & {:keys [from to]}]
  (let [sequence (:sim/sequence sim)
        delivery {:at (+ (:sim/time sim) (tick-delay delay))
                  :sequence sequence
                  :target target
                  :event event
                  :from from
                  :to to}]
    (-> sim
        (update :sim/sequence inc)
        (update :sim/events conj delivery)
        (update :sim/events #(vec (sort-by (juxt :at :sequence) %))))))

(defn- trace
  [sim value]
  (update sim :sim/trace conj (assoc value :time (:sim/time sim))))

(defn- blocked?
  [sim from to]
  (contains? (:sim/blocked-links sim) [from to]))

(declare dispatch-router dispatch-worker)

(defn- worker-operation-delay
  [effect]
  (let [candidate (:assignment/candidate effect)
        cached (:estimate/cached-token-count candidate)
        missing (:estimate/missing-token-count candidate)
        tier (:candidate/cache-tier candidate)]
    (case (:effect/op effect)
      :worker/restore-prefix
      (+ (max (:candidate/queue-ms candidate)
              (:candidate/prefix-load-ms candidate))
         (:candidate/gpu-restore-ms candidate))

      :worker/prefill-suffix
      (+ (if (or (= :gpu tier) (zero? cached))
           (max (:candidate/queue-ms candidate)
                (:candidate/prefix-load-ms candidate))
           0)
         (* missing (:candidate/prefill-ms-per-token candidate)))

      :worker/decode (:candidate/first-token-ms candidate)
      0)))

(defn- simulated-tokens
  [effect]
  (let [request (:assignment/request effect)]
    (or (:request/simulated-output request)
        (mapv (fn [offset]
                (+ 1000000 (long offset)))
              (range (:request/max-new-tokens request))))))

(defn- interpret-router-effect
  [sim effect]
  (case (:effect/op effect)
    :router/send-offer
    (schedule sim (get-in sim [:sim/config :network-delay])
              [:worker (:effect/to effect)]
              {:event/type :assignment/offered
               :assignment/id (:assignment/id effect)
               :assignment/worker-epoch (:assignment/worker-epoch effect)
               :assignment/request (:assignment/request effect)
               :assignment/candidate (:assignment/candidate effect)}
              :from :router :to (:effect/to effect))

    :router/set-offer-timer
    (schedule sim (get-in sim [:sim/config :offer-timeout]) [:router]
              {:event/type :router/offer-timeout
               :request/id (:request/id effect)
               :assignment/id (:assignment/id effect)})

    :router/cancel-offer-timer
    ;; Timer events are immutable. The assignment fence makes their later
    ;; delivery a no-op and avoids simulator-only cancellation semantics.
    sim

    :router/send-cancel
    (schedule sim (get-in sim [:sim/config :network-delay])
              [:worker (:effect/to effect)]
              {:event/type :assignment/cancelled
               :request/id (:request/id effect)
               :assignment/id (:assignment/id effect)}
              :from :router :to (:effect/to effect))

    :router/deliver
    (-> sim
        (update :sim/responses conj (dissoc effect :effect/op))
        (trace {:trace/type :response :response effect}))

    (throw (ex-info "Simulator cannot interpret router effect" {:effect effect}))))

(defn- interpret-worker-effect
  [sim worker-id effect]
  (case (:effect/op effect)
    :worker/send-offer-result
    (schedule sim (get-in sim [:sim/config :network-delay]) [:router]
              {:event/type :worker/offer-result
               :request/id (:request/id effect)
               :assignment/id (:assignment/id effect)
               :event/accepted? (:event/accepted? effect)
               :event/reason (:event/reason effect)}
              :from worker-id :to :router)

    :worker/send-result
    (schedule sim (get-in sim [:sim/config :network-delay]) [:router]
              {:event/type :worker/result
               :request/id (:request/id effect)
               :assignment/id (:assignment/id effect)
               :event/result (:event/result effect)}
              :from worker-id :to :router)

    :worker/restore-prefix
    (schedule sim (worker-operation-delay effect) [:worker worker-id]
              {:event/type :worker/restore-result
               :assignment/id (:assignment/id effect)
               :event/ok? true})

    :worker/prefill-suffix
    (schedule sim (worker-operation-delay effect) [:worker worker-id]
              {:event/type :worker/prefill-result
               :assignment/id (:assignment/id effect)
               :event/ok? true})

    :worker/decode
    (schedule sim (worker-operation-delay effect) [:worker worker-id]
              {:event/type :worker/decode-result
               :assignment/id (:assignment/id effect)
               :event/ok? true
               :event/tokens (simulated-tokens effect)})

    :worker/cancel-operation sim

    (throw (ex-info "Simulator cannot interpret worker effect" {:effect effect}))))

(defn- dispatch-router
  [sim event]
  (let [{next-state :state effects :effects}
        (router/transition (:sim/router sim) event)]
    (reduce interpret-router-effect
            (-> sim
                (assoc :sim/router next-state)
                (trace {:trace/type :router/event :event event :effects effects}))
            effects)))

(defn- dispatch-worker
  [sim worker-id event]
  (if-let [state (get-in sim [:sim/workers worker-id])]
    (let [{next-state :state effects :effects} (worker/transition state event)]
      (reduce #(interpret-worker-effect %1 worker-id %2)
              (-> sim
                  (assoc-in [:sim/workers worker-id] next-state)
                  (trace {:trace/type :worker/event :worker/id worker-id
                          :event event :effects effects}))
              effects))
    (trace sim {:trace/type :missing-worker :worker/id worker-id :event event})))

(defn submit
  "Submit a request and request-specific candidates to the simulated router."
  [sim request candidates]
  (dispatch-router sim {:event/type :request/submitted
                        :event/request request
                        :event/candidates candidates}))

(defn cancel
  "Cancel `request-id` at the simulated consumer boundary."
  [sim request-id]
  (dispatch-router sim {:event/type :request/cancelled
                        :request/id request-id}))

(defn- deliver-event
  [sim {:keys [target event from to] :as delivery}]
  (cond
    (and from to (blocked? sim from to))
    (trace sim {:trace/type :network/drop :delivery delivery})

    (= target [:router])
    (dispatch-router sim event)

    (= :worker (first target))
    (let [worker-id (second target)]
      (if (contains? (:sim/paused sim) worker-id)
        (-> sim
            (update :sim/events conj (assoc delivery :at (inc (:sim/time sim))))
            (update :sim/events #(vec (sort-by (juxt :at :sequence) %))))
        (dispatch-worker sim worker-id event)))

    :else
    (throw (ex-info "Simulation event has an unknown target"
                    {:delivery delivery}))))

(defn- drain-current
  [sim]
  (loop [sim sim
         steps 0]
    (when (> steps 10000)
      (throw (ex-info "Simulation failed to quiesce at one logical tick"
                      {:time (:sim/time sim)})))
    (if-let [event (first (filter #(<= (:at %) (:sim/time sim))
                                  (:sim/events sim)))]
      (recur (deliver-event (update sim :sim/events
                                    (fn [events]
                                      (vec (remove #(= (:sequence %) (:sequence event))
                                                   events))))
                            event)
             (inc steps))
      sim)))

(defn advance
  "Advance logical time by one tick and deliver every event now due."
  [sim]
  (drain-current (update sim :sim/time inc)))

(defn run
  "Advance at most `ticks` logical ticks."
  [sim ticks]
  (when-not (and (integer? ticks) (not (neg? ticks)))
    (throw (ex-info "Simulation tick count must be non-negative" {:ticks ticks})))
  (nth (iterate advance sim) ticks))

(defn run-until-response
  "Advance until `request-id` has a response or `max-ticks` is exhausted."
  [sim request-id max-ticks]
  (loop [sim sim
         remaining (long max-ticks)]
    (if (or (some #(= request-id (:request/id %)) (:sim/responses sim))
            (zero? remaining))
      sim
      (recur (advance sim) (dec remaining)))))

(defn response
  "Return the latest simulated response for `request-id`, or nil."
  [sim request-id]
  (last (filter #(= request-id (:request/id %)) (:sim/responses sim))))

(defn partition-link
  "Drop future deliveries on the directed link `from` → `to`."
  [sim from to]
  (update sim :sim/blocked-links conj [from to]))

(defn heal
  "Restore every simulated directed network link."
  [sim]
  (assoc sim :sim/blocked-links #{}))

(defn pause
  "Pause a worker while retaining its volatile state and queued deliveries."
  [sim worker-id]
  (update sim :sim/paused conj worker-id))

(defn resume
  "Resume a paused worker."
  [sim worker-id]
  (update sim :sim/paused disj worker-id))

(defn crash
  "Crash a worker, lose volatile assignments, increment its epoch, and notify
  the router so affected requests can fall back to another candidate."
  [sim worker-id]
  (let [assignments (->> (get-in sim [:sim/workers worker-id
                                      :worker/assignments])
                         vals
                         (remove #(contains? worker/terminal-phases
                                             (:assignment/phase %)))
                         (sort-by (comp pr-str :assignment/id)))
        sim (dispatch-worker sim worker-id {:event/type :worker/crashed})]
    (reduce (fn [current assignment]
              (dispatch-router
               current
               {:event/type :worker/unavailable
                :worker/id worker-id
                :request/id (get-in assignment [:assignment/request :request/id])
                :assignment/id (:assignment/id assignment)}))
            sim assignments)))

(defn restart
  "Restart a crashed worker with its incremented epoch."
  [sim worker-id]
  (dispatch-worker sim worker-id {:event/type :worker/restarted}))

(defn quiet?
  "Return true when the simulator has no scheduled work."
  [sim]
  (empty? (:sim/events sim)))
