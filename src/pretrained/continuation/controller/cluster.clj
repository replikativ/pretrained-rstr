(ns pretrained.continuation.controller.cluster
  "Stateful interpreter for the pure cluster router.

  Candidate discovery remains an injected control-plane concern. This runtime
  owns request identity, fenced offer timeouts, worker messages, and delivery to
  a consumer adapter such as an OpenAI-compatible HTTP stream."
  (:require [pretrained.continuation.controller.router :as router])
  (:import (java.io Closeable)
           (java.util.concurrent Executors ScheduledExecutorService TimeUnit)))

(defrecord ClusterController
    [machine timers send! deliver! schedule! cancel-timer! close-scheduler!
     offer-timeout-ms closed?]
  Closeable
  (close [this]
    (locking this
      (when (compare-and-set! closed? false true)
        (doseq [[_ timer] @timers]
          (cancel-timer! timer))
        (reset! timers {})
        (close-scheduler!)))))

(defn- default-scheduler
  []
  (let [scheduler (Executors/newSingleThreadScheduledExecutor)]
    {:schedule!
     (fn [delay-ms task]
       (.schedule ^ScheduledExecutorService scheduler
                  ^Runnable (reify Runnable (run [_] (task)))
                  (long delay-ms) TimeUnit/MILLISECONDS))
     :cancel-timer! (fn [timer] (when timer (.cancel timer false)))
     :close-scheduler! #(.close ^ScheduledExecutorService scheduler)}))

(defn open-controller
  "Open a cluster request controller.

  `opts` requires `:send!`, which receives router network effects, and
  `:deliver!`, which receives terminal consumer effects. `:offer-timeout-ms`
  defaults to 100. Tests and external event loops may replace the scheduler by
  supplying `:schedule!`, `:cancel-timer!`, and `:close-scheduler!` together."
  [{:keys [send! deliver! offer-timeout-ms schedule! cancel-timer!
           close-scheduler!]
    :or {offer-timeout-ms 100}}]
  (when-not (and (ifn? send!) (ifn? deliver!))
    (throw (ex-info "Cluster controller requires send and delivery callbacks" {})))
  (when-not (and (integer? offer-timeout-ms) (pos? offer-timeout-ms))
    (throw (ex-info "Offer timeout must be a positive integer"
                    {:offer-timeout-ms offer-timeout-ms})))
  (let [custom? (or schedule! cancel-timer! close-scheduler!)]
    (when (and custom?
               (not (every? ifn? [schedule! cancel-timer! close-scheduler!])))
      (throw (ex-info "Custom scheduling requires schedule, cancel, and close functions"
                      {})))
    (let [scheduling (if custom?
                       {:schedule! schedule! :cancel-timer! cancel-timer!
                        :close-scheduler! close-scheduler!}
                       (default-scheduler))]
      (map->ClusterController
       (merge scheduling
              {:machine (atom (router/initial-state))
               :timers (atom {})
               :send! send!
               :deliver! deliver!
               :offer-timeout-ms (long offer-timeout-ms)
               :closed? (atom false)})))))

(defn state
  "Return the current immutable router state."
  [controller]
  @(:machine controller))

(declare handle-event!)

(defn- timer-key
  [effect]
  [(:request/id effect) (:assignment/id effect)])

(defn- interpret-effect!
  [controller effect]
  (case (:effect/op effect)
    :router/set-offer-timer
    (let [key (timer-key effect)
          timer ((:schedule! controller)
                 (:offer-timeout-ms controller)
                 #(do
                    (swap! (:timers controller) dissoc key)
                    (handle-event!
                     controller
                     {:event/type :router/offer-timeout
                      :request/id (:request/id effect)
                      :assignment/id (:assignment/id effect)})))]
      (swap! (:timers controller) assoc key timer))

    :router/cancel-offer-timer
    (let [key (timer-key effect)]
      (when-let [timer (get @(:timers controller) key)]
        ((:cancel-timer! controller) timer))
      (swap! (:timers controller) dissoc key))

    :router/deliver
    ((:deliver! controller) effect)

    ((:send! controller) effect)))

(defn handle-event!
  "Apply one consumer, worker, or timeout event and interpret its effects."
  [controller event]
  (when @(:closed? controller)
    (throw (ex-info "Cluster controller is closed" {:event event})))
  (let [{next-state :state effects :effects}
        (locking controller
          (let [transition (router/transition @(:machine controller) event)]
            (reset! (:machine controller) (:state transition))
            transition))]
    (doseq [effect effects]
      (interpret-effect! controller effect))
    next-state))

(defn submit!
  "Submit a validated generation request and its request-specific candidates."
  [controller request candidates]
  (handle-event! controller
                 {:event/type :request/submitted
                  :event/request request
                  :event/candidates candidates}))

(defn cancel-request!
  "Cancel `request-id`; active worker work is fenced and asked to stop."
  [controller request-id]
  (handle-event! controller
                 {:event/type :request/cancelled
                  :request/id request-id}))
