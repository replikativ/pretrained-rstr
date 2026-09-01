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
     closed?]
  Closeable
  (close [_]
    (when (compare-and-set! closed? false true)
      (doseq [[_ task] @tasks]
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
  serializes GPU-facing operations. Close the returned controller when done."
  [pool worker-opts {:keys [handlers send! submit! cancel! close-submit!]
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
               :closed? (atom false)})))))

(defn state
  "Return the current immutable worker-machine state."
  [controller]
  @(:machine controller))

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
                snapshot (page-pool/residency-snapshot (:pool controller))
                plan (residency/plan-capacity-admission
                      snapshot continuation-id target)]
            (if-not (:admissible? plan)
              {:state machine
               :effects [(decline-effect machine event :insufficient-pages)]}
              (let [required (:required-pages plan)
                    observed (assoc machine
                                    :worker/free-pages required
                                    :worker/evictable-pages 0)
                    transition (worker/transition observed event)]
                (if-not (accepted? (:effects transition))
                  transition
                  (let [admission (residency/reserve-admission!
                                   (:pool controller) continuation-id target)
                        reservation (:capacity-reservation admission)]
                    (swap! (:reservations controller)
                           assoc assignment-id reservation)
                    transition))))))))))

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
        handle ((:submit! controller) task)]
    (swap! (:tasks controller)
           #(if (contains? % task-key) (assoc % task-key handle) %))))

(defn- interpret-effects!
  [controller effects]
  (doseq [effect effects]
    (cond
      (contains? operation-result-types (:effect/op effect))
      (submit-operation! controller effect)

      (= :worker/cancel-operation (:effect/op effect))
      (when-not (assignment-running? controller (:assignment/id effect))
        (release-assignment! controller (:assignment/id effect)))

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
