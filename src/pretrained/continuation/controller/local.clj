(ns pretrained.continuation.controller.local
  "Stateful worker-local interpreter for the pure continuation protocol.

  This namespace is the authority boundary around a real GPU page pool. It
  revalidates offers, atomically reserves projected prompt-plus-generation
  capacity, and invokes injected restore, prefill, and decode handlers. Network
  transports only see immutable worker effects; tensor data never crosses this
  boundary through Kabel."
  (:require [pretrained.continuation.controller.router :as router]
            [pretrained.continuation.controller.worker :as worker]
            [pretrained.continuation.page-pool :as page-pool]
            [pretrained.continuation.residency :as residency])
  (:import (java.io Closeable)
           (java.util UUID)
           (java.util.concurrent ExecutorService Executors Future)))

(def ^:private operation-result-types
  {:worker/restore-prefix :worker/restore-result
   :worker/prefill-suffix :worker/prefill-result
   :worker/decode :worker/decode-result})

(defrecord LocalController
    [pool machine reservations tasks handlers send! submit! cancel! close-submit!
     cancel-operation! closed?]
  Closeable
  (close [_]
    (when (compare-and-set! closed? false true)
      (doseq [[[assignment-id _] task] @tasks]
        (when cancel-operation!
          (cancel-operation! {:effect/op :worker/cancel-operation
                              :assignment/id assignment-id}))
        (cancel! task))
      (close-submit!)
      (reset! tasks {})
      (doseq [[_ reservation] @reservations]
        (page-pool/release-capacity! pool reservation))
      (reset! reservations {}))))

(defn- default-submitter
  []
  (let [executor (Executors/newSingleThreadExecutor)]
    {:submit! (fn [task]
                (.submit ^ExecutorService executor
                         ^Runnable (reify Runnable (run [_] (task)))))
     :cancel! (fn [task]
                (when task (.cancel ^Future task false)))
     :close-submit! #(.close ^ExecutorService executor)}))

(defn open-controller
  "Open a worker-local controller around `pool`.

  `worker-opts` follows `worker/initial-state`. `opts` requires `:handlers`, a
  map from `:worker/restore-prefix`, `:worker/prefill-suffix`, and
  `:worker/decode` to one-argument functions. Restore and prefill handlers may
  return nil or `{:ok? boolean :reason value}`; decode additionally returns
  `:tokens`. `:send!` receives immutable protocol effects.

  Tests and event-loop integrations may provide the complete trio `:submit!`,
  `:cancel!`, and `:close-submit!`; otherwise a private single-thread executor
  serializes GPU-facing operations. `:cancel-operation!` may additionally fence
  work owned by an external runtime. Close the returned controller when done."
  [pool worker-opts {:keys [handlers send! submit! cancel! close-submit!
                            cancel-operation!]
                     :or {send! (constantly nil)}}]
  (when-not (and (map? handlers)
                 (every? #(ifn? (get handlers %))
                         (keys operation-result-types)))
    (throw (ex-info "Local controller requires every operation handler"
                    {:required (set (keys operation-result-types))
                     :provided (set (keys handlers))})))
  (let [custom? (or submit! cancel! close-submit!)]
    (when (and custom? (not (every? ifn? [submit! cancel! close-submit!])))
      (throw (ex-info "Custom submission requires submit, cancel, and close functions"
                      {})))
    (when-not (or (nil? cancel-operation!) (ifn? cancel-operation!))
      (throw (ex-info "Operation cancellation callback must be callable" {})))
    (let [submission (if custom?
                       {:submit! submit! :cancel! cancel!
                        :close-submit! close-submit!}
                       (default-submitter))]
      (map->LocalController
       (merge submission
              {:pool pool
               :machine (atom (worker/initial-state worker-opts))
               :reservations (atom {})
               :tasks (atom {})
               :handlers handlers
               :send! send!
               :cancel-operation! cancel-operation!
               :closed? (atom false)})))))

(defn state
  "Return the current immutable worker-machine state."
  [controller]
  @(:machine controller))

(defn observation
  "Return an immutable scheduling observation for candidate derivation.

  `measurements` supplies `:worker/node`, queue and model-performance costs,
  context capacity, device upload throughput, lower-tier throughputs/fixed
  costs, and object-store availability. A deterministic simulator may provide
  `:worker/transfer-capabilities`; a real worker derives them from its page-pool
  session. Identity, epoch, loaded models, online
  state, page geometry, current free pages, recoverable eviction capacity, and
  exact GPU prefixes are read authoritatively from the controller and pool.

  A route is advertised as an exact GPU prefix when its `:cache/policy` contains
  `:model-fingerprint` and `:prefix-hash`. The paged controller handlers install
  these fields after decode."
  [controller measurements]
  (let [machine @(:machine controller)
        snapshot (page-pool/residency-snapshot (:pool controller))
        protected
        (into #{}
              (keep (fn [[_ assignment]]
                      (when-not (contains? worker/terminal-phases
                                           (:assignment/phase assignment))
                        (let [request (:assignment/request assignment)]
                          (or (:request/continuation-id request)
                              (:request/id request))))))
              (:worker/assignments machine))
        gpu-prefixes
        (into {}
              (keep
               (fn [[continuation-id resident-route]]
                 (let [policy (:cache/policy resident-route)
                       model (:model-fingerprint policy)
                       prefix (:prefix-hash policy)]
                   (when (and (string? model) (uuid? prefix))
                     [[model prefix]
                      {:continuation-id continuation-id
                       :token-count (:token-count resident-route)
                       :bytes (or (:bytes policy)
                                  (page-pool/route-bytes
                                   (:pool controller) continuation-id))}]))))
              (:routes snapshot))]
    (merge measurements
           {:worker/id (:worker/id machine)
            :worker/epoch (:worker/epoch machine)
            :worker/models (:worker/models machine)
            :worker/online? (:worker/online? machine)
            :worker/page-size (:page-size snapshot)
            :worker/free-pages (count (:free-pages snapshot))
            :worker/evictable-pages
            (residency/evictable-page-count
             snapshot {:protected-continuation-ids protected})
            :worker/transfer-capabilities
            (or (:worker/transfer-capabilities measurements)
                (page-pool/transfer-capabilities (:pool controller)))
            :worker/gpu-prefixes gpu-prefixes})))

(defn- decline-effect
  [machine event reason]
  {:effect/op :worker/send-offer-result
   :effect/to :router
   :request/id (get-in event [:assignment/request :request/id])
   :assignment/id (:assignment/id event)
   :assignment/worker-epoch (:worker/epoch machine)
   :event/accepted? false
   :event/reason reason})

(defn- accepted?
  [effects]
  (some #(and (= :worker/send-offer-result (:effect/op %))
              (:event/accepted? %))
        effects))

(defn- ensure-gpu-prefix!
  [pool continuation-id candidate]
  (let [cached (:estimate/cached-token-count candidate)]
    (if-not (and (= :gpu (:candidate/cache-tier candidate)) (pos? cached))
      {:ok? true :forked? false}
      (let [source-id (or (:candidate/source-continuation-id candidate)
                          continuation-id)
            source (page-pool/route pool source-id)
            target (page-pool/route pool continuation-id)]
        (cond
          (and target (= cached (:token-count target)))
          {:ok? true :forked? false}

          target
          {:ok? false :reason :resident-prefix-mismatch}

          (and source (= cached (:token-count source)))
          (do
            (page-pool/fork-route! pool source-id continuation-id)
            {:ok? true :forked? true})

          :else
          {:ok? false :reason :resident-prefix-unavailable})))))

(defn- offer-transition!
  [controller event]
  (locking controller
    (let [machine @(:machine controller)
          assignment-id (:assignment/id event)
          duplicate? (contains? (:worker/assignments machine) assignment-id)]
      (if duplicate?
        (worker/transition machine event)
        (locking (:pool controller)
          (let [request (:assignment/request event)
                candidate (router/estimate-candidate
                           request (:assignment/candidate event))
                continuation-id (or (:request/continuation-id request)
                                    (:request/id request))
                target (:estimate/target-token-count candidate)
                provisional
                (worker/transition
                 (assoc machine
                        :worker/free-pages (:estimate/required-pages candidate)
                        :worker/evictable-pages 0)
                 event)]
            (if-not (accepted? (:effects provisional))
              provisional
              (let [{:keys [ok? forked? reason]}
                    (ensure-gpu-prefix!
                     (:pool controller) continuation-id candidate)]
                (if-not ok?
                  {:state machine
                   :effects [(decline-effect machine event reason)]}
                  (let [snapshot (page-pool/residency-snapshot
                                  (:pool controller))
                        plan (residency/plan-capacity-admission
                              snapshot continuation-id target)]
                    (if-not (:admissible? plan)
                      (do
                        (when forked?
                          (page-pool/release-route!
                           (:pool controller) continuation-id))
                        {:state machine
                         :effects [(decline-effect
                                    machine event :insufficient-pages)]})
                      (let [admission (residency/reserve-admission!
                                       (:pool controller) continuation-id target)
                            reservation (:capacity-reservation admission)]
                        (swap! (:reservations controller)
                               assoc assignment-id reservation)
                        provisional))))))))))))

(declare handle-event!)

(defn- result-event
  [effect result]
  (merge {:event/type (get operation-result-types (:effect/op effect))
          :assignment/id (:assignment/id effect)
          :event/ok? (not (false? (:ok? result)))}
         (select-keys result [:reason :tokens])
         (when (contains? result :reason)
           {:event/reason (:reason result)})
         (when (contains? result :tokens)
           {:event/tokens (:tokens result)})))

(defn- release-assignment!
  [controller assignment-id]
  (when-let [reservation (get @(:reservations controller) assignment-id)]
    (page-pool/release-capacity! (:pool controller) reservation)
    (swap! (:reservations controller) dissoc assignment-id)))

(defn- assignment-running?
  [controller assignment-id]
  (some #(= assignment-id (first %)) (keys @(:tasks controller))))

(defn- submit-operation!
  [controller effect]
  (let [assignment-id (:assignment/id effect)
        task-key [assignment-id (UUID/randomUUID)]
        reservation (get @(:reservations controller) assignment-id)
        operation (assoc effect :worker/capacity-reservation reservation)
        task (fn []
               (try
                 (let [result (or ((get (:handlers controller) (:effect/op effect))
                                   operation)
                                  {})
                       _ (when-not (map? result)
                           (throw (ex-info "Operation handler must return a map or nil"
                                           {:effect/op (:effect/op effect)
                                            :result result})))]
                   (handle-event! controller (result-event effect result)))
                 (catch Throwable error
                   (when-not @(:closed? controller)
                     (handle-event!
                      controller
                      (result-event effect
                                    {:ok? false
                                     :reason {:type :operation-failed
                                              :message (.getMessage error)}}))))
                 (finally
                   (swap! (:tasks controller) dissoc task-key)
                   (when (and (not (assignment-running? controller assignment-id))
                              (nil? (get-in @(:machine controller)
                                            [:worker/assignments assignment-id])))
                     (release-assignment! controller assignment-id)))))
        _ (swap! (:tasks controller) assoc task-key nil)
        handle ((:submit! controller) task)
        installed? (volatile! false)]
    (swap! (:tasks controller)
           #(if (contains? % task-key)
              (do (vreset! installed? true)
                  (assoc % task-key handle))
              %))
    (when (and @installed?
               handle
               (nil? (get-in @(:machine controller)
                             [:worker/assignments assignment-id])))
      (when-let [cancel-operation! (:cancel-operation! controller)]
        (cancel-operation! {:effect/op :worker/cancel-operation
                            :assignment/id assignment-id}))
      ((:cancel! controller) handle))))

(defn- interpret-effects!
  [controller effects]
  (doseq [effect effects]
    (cond
      (contains? operation-result-types (:effect/op effect))
      (submit-operation! controller effect)

      (= :worker/cancel-operation (:effect/op effect))
      (let [assignment-id (:assignment/id effect)
            matching (for [[[task-assignment-id _] task] @(:tasks controller)
                           :when (= assignment-id task-assignment-id)]
                       task)]
        (when-let [cancel-operation! (:cancel-operation! controller)]
          (cancel-operation! effect))
        (doseq [task matching :when task]
          ((:cancel! controller) task))
        (when-not (assignment-running? controller assignment-id)
          (release-assignment! controller assignment-id)))

      (= :worker/send-result (:effect/op effect))
      (do
        (release-assignment! controller (:assignment/id effect))
        ((:send! controller) effect))

      :else
      ((:send! controller) effect))))

(defn handle-event!
  "Apply one offered-assignment, operation-result, cancellation, or lifecycle event.

  Offer handling is serialized with the page pool: semantic validation happens
  before any eviction, and an accepted reply is emitted only after projected
  capacity has been reserved. Returns the updated immutable worker state."
  [controller event]
  (when @(:closed? controller)
    (throw (ex-info "Local controller is closed" {:event event})))
  (let [{next-state :state effects :effects}
        (locking controller
          (let [transition
                (if (= :assignment/offered (:event/type event))
                  (offer-transition! controller event)
                  (worker/transition @(:machine controller) event))]
            (reset! (:machine controller) (:state transition))
            transition))]
    (interpret-effects! controller effects)
    next-state))
