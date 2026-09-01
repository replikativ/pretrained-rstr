(ns pretrained.continuation.controller.kabel
  "Optional live Kabel transport for the continuation controller.

  This namespace is loaded with the `:distributed-demo` alias. It carries only
  observations, offers, acknowledgements, cancellations, and results. Datahike
  remains the durable fact plane and immutable tensor chunks remain on their
  Konserve/tiered-store path."
  (:require [clojure.core.async :as async]
            [pretrained.continuation.controller.candidates :as candidates]
            [pretrained.continuation.controller.cluster :as cluster]
            [pretrained.continuation.controller.discovery :as discovery]
            [pretrained.continuation.controller.local :as local]
            [pretrained.continuation.controller.wire :as wire])
  (:import (java.io Closeable)
           (java.util UUID)
           (java.util.concurrent Executors RejectedExecutionException
                                 ScheduledExecutorService TimeUnit)))

(def ^:private active-router-phases #{:offered :assigned})
(def ^:private worker-message-types
  #{:continuation/offer :continuation/cancel})
(def ^:private router-message-types
  #{:continuation/offer-result
    :continuation/result
    :continuation/worker-unavailable})

(defn- scheduler
  []
  (Executors/newSingleThreadScheduledExecutor))

(defn- close-scheduler!
  [^ScheduledExecutorService executor]
  (.shutdownNow executor))

(defn- report-error!
  [on-error! error context]
  (on-error! (ex-info "Continuation Kabel endpoint failed" context error)))

(declare expire-worker!)

(defrecord RouterEndpoint
    [database discovery routes expiry-timers controller heartbeat-timeout-ms
     chunk-size scheduler on-error! closed?]
  Closeable
  (close [_]
    (when (compare-and-set! closed? false true)
      (doseq [[_ timer] @expiry-timers]
        (.cancel timer false))
      (reset! expiry-timers {})
      (.close ^Closeable controller)
      (close-scheduler! scheduler)
      (reset! routes {})
      (reset! discovery (discovery/initial-state)))))

(defn- unavailable-active-assignments!
  [endpoint worker-id]
  (doseq [[request-id assignment]
          (:router/requests (cluster/state (:controller endpoint)))
          :when (and (contains? active-router-phases
                                (:assignment/phase assignment))
                     (= worker-id
                        (get-in assignment
                                [:assignment/candidate
                                 :candidate/worker-id])))]
    (cluster/handle-event!
     (:controller endpoint)
     {:event/type :worker/unavailable
      :worker/id worker-id
      :request/id request-id
      :assignment/id (:assignment/id assignment)})))

(defn- schedule-unavailable!
  [endpoint worker-id]
  (when-not @(:closed? endpoint)
    (try
      (.execute
       ^ScheduledExecutorService (:scheduler endpoint)
       ^Runnable
       (reify Runnable
         (run [_]
           (when-not @(:closed? endpoint)
             (unavailable-active-assignments! endpoint worker-id)))))
      (catch RejectedExecutionException _
        nil))))

(defn- detach-worker!
  [endpoint worker-id connection-id]
  (let [removed?
        (locking endpoint
          (when (= connection-id
                   (get-in @(:routes endpoint) [worker-id :connection-id]))
            (swap! (:routes endpoint) dissoc worker-id)
            (when-let [timer (get @(:expiry-timers endpoint) worker-id)]
              (.cancel timer false))
            (swap! (:expiry-timers endpoint) dissoc worker-id)
            (swap! (:discovery endpoint) discovery/remove-worker worker-id)
            true))]
    (when removed?
      (schedule-unavailable! endpoint worker-id))))

(defn- expire-worker!
  [endpoint worker-id connection-id version]
  (locking endpoint
    (let [current (get-in @(:discovery endpoint)
                          [:discovery/workers worker-id])]
      (when (and (= connection-id
                    (get-in @(:routes endpoint) [worker-id :connection-id]))
                 (= version
                    [(:worker/epoch current) (:worker/sequence current)]))
        (detach-worker! endpoint worker-id connection-id)))))

(defn- observe-worker!
  [endpoint connection-id out observation]
  (let [observation (candidates/worker-observation observation)
        worker-id (:worker/id observation)]
    (locking endpoint
      (let [next-state (discovery/observe @(:discovery endpoint) observation)
            accepted (get-in next-state [:discovery/workers worker-id])]
        (when (= accepted observation)
          (reset! (:discovery endpoint) next-state)
          (swap! (:routes endpoint) assoc worker-id
                 {:connection-id connection-id :out out})
          (when-let [timer (get @(:expiry-timers endpoint) worker-id)]
            (.cancel timer false))
          (let [version [(:worker/epoch accepted) (:worker/sequence accepted)]
                timer (.schedule
                       ^ScheduledExecutorService (:scheduler endpoint)
                       ^Runnable
                       (reify Runnable
                         (run [_]
                           (when-not @(:closed? endpoint)
                             (expire-worker! endpoint worker-id
                                             connection-id version))))
                       (long (:heartbeat-timeout-ms endpoint))
                       TimeUnit/MILLISECONDS)]
            (swap! (:expiry-timers endpoint) assoc worker-id timer)))))))

(defn- send-router-effect!
  [endpoint effect]
  (let [message (wire/effect->message effect)
        worker-id (:message/to message)]
    (if-let [{:keys [connection-id out]}
             (get @(:routes endpoint) worker-id)]
      (when-not (async/put! out message)
        (detach-worker! endpoint worker-id connection-id))
      (when (= :router/send-offer (:effect/op effect))
        (schedule-unavailable! endpoint worker-id)))))

(defn open-router-endpoint
  "Open a live router endpoint around a Datahike database or connection.

  Options require `:deliver!`, the terminal consumer callback. Heartbeat expiry
  defaults to 3 seconds, offer timeout to 100 ms, and candidate chunk size to
  the candidate planner default. `:on-error!` receives asynchronous middleware
  failures. Close the returned endpoint when the server stops."
  [database {:keys [deliver! heartbeat-timeout-ms offer-timeout-ms chunk-size
                    on-error!]
             :or {heartbeat-timeout-ms 3000
                  offer-timeout-ms 100
                  chunk-size 256
                  on-error! (fn [error] (.printStackTrace ^Throwable error))}}]
  (when-not (ifn? deliver!)
    (throw (ex-info "Router endpoint requires a delivery callback" {})))
  (when-not (and (integer? heartbeat-timeout-ms)
                 (pos? heartbeat-timeout-ms))
    (throw (ex-info "Heartbeat timeout must be a positive integer"
                    {:heartbeat-timeout-ms heartbeat-timeout-ms})))
  (when-not (and (integer? chunk-size) (pos? chunk-size))
    (throw (ex-info "Controller chunk size must be a positive integer"
                    {:chunk-size chunk-size})))
  (when-not (ifn? on-error!)
    (throw (ex-info "Router endpoint error handler must be callable" {})))
  (let [holder (atom nil)
        controller (cluster/open-controller
                    {:send! #(send-router-effect! @holder %)
                     :deliver! deliver!
                     :offer-timeout-ms offer-timeout-ms})
        endpoint (->RouterEndpoint
                  database (atom (discovery/initial-state)) (atom {}) (atom {})
                  controller (long heartbeat-timeout-ms) (long chunk-size)
                  (scheduler) on-error! (atom false))]
    (reset! holder endpoint)
    endpoint))

(defn observations
  "Return the router's currently live worker observations in stable order."
  [endpoint]
  (discovery/observations @(:discovery endpoint)))

(defn router-state
  "Return the live endpoint's immutable cluster-router state."
  [endpoint]
  (cluster/state (:controller endpoint)))

(defn submit!
  "Derive candidates from live observations and submit one generation request.

  The endpoint database may be a Datahike connection or an immutable database
  value. Returns the updated router state."
  [endpoint request]
  (let [database (:database endpoint)
        database (if (instance? clojure.lang.IDeref database)
                   @database
                   database)
        ranked (candidates/candidates
                database request (observations endpoint)
                {:chunk-size (:chunk-size endpoint)})]
    (cluster/submit! (:controller endpoint) request ranked)))

(defn cancel-request!
  "Cancel an active request and fence any late worker result."
  [endpoint request-id]
  (cluster/cancel-request! (:controller endpoint) request-id))

(defn router-middleware
  "Return Kabel middleware that attaches one worker connection to `endpoint`.

  Recognized continuation messages are consumed. All unrelated messages pass
  through unchanged, allowing this middleware to share a peer with Datahike,
  Konserve Sync, and distributed-scope."
  [endpoint]
  (fn [[supervisor peer [in out]]]
    (let [next-in (async/chan 64)
          next-out (async/chan 64)
          connection-id (UUID/randomUUID)
          workers (atom #{})]
      (async/go
        (try
          (loop []
            (when-some [message (async/<! in)]
              (if (wire/control-message? message)
                (cond
                  (= :continuation/worker-observation (:type message))
                  (when-let [observation (wire/message->observation message)]
                    (swap! workers conj (:worker/id observation))
                    (observe-worker! endpoint connection-id out observation))

                  (contains? router-message-types (:type message))
                  (when-let [event (wire/router-event message)]
                    (cluster/handle-event! (:controller endpoint) event))

                  :else
                  (throw (ex-info "Router received a worker-incompatible message"
                                  {:type (:type message)})))
                (async/>! next-in message))
              (recur)))
          (catch Throwable error
            (report-error! (:on-error! endpoint) error
                           {:endpoint :router
                            :connection-id connection-id}))
          (finally
            (doseq [worker-id @workers]
              (detach-worker! endpoint worker-id connection-id))
            (async/close! next-in))))
      (async/go-loop []
        (when-some [message (async/<! next-out)]
          (async/>! out message)
          (recur)))
      [supervisor peer [next-in next-out]])))

(defrecord WorkerEndpoint
    [controller measurements sequence heartbeat-ms scheduler heartbeat-task
     connection out on-error! closed?]
  Closeable
  (close [_]
    (when (compare-and-set! closed? false true)
      (when-let [task @heartbeat-task]
        (.cancel task false))
      (close-scheduler! scheduler)
      (.close ^Closeable controller)
      (reset! out nil)
      (reset! connection nil))))

(defn- publish-observation!
  [endpoint]
  (when-let [out @(:out endpoint)]
    (let [measurements (:measurements endpoint)
          measurements (if (map? measurements)
                         measurements
                         (measurements))
          sequence (swap! (:sequence endpoint) inc)
          observation (assoc (local/observation (:controller endpoint)
                                                measurements)
                             :worker/sequence sequence)]
      (async/put! out (wire/observation->message observation)))))

(defn- send-worker-effect!
  [endpoint effect]
  (when-let [out @(:out endpoint)]
    (async/put! out (wire/effect->message effect))))

(defn open-worker-endpoint
  "Open a worker-local controller with periodic Kabel observations.

  `pool`, `worker-opts`, and the handler contract match
  `local/open-controller`. Options require `:handlers` and `:measurements` (a
  map or zero-argument function). Heartbeats default to one second. The
  endpoint owns and closes its local controller."
  [pool worker-opts {:keys [handlers measurements heartbeat-ms on-error!]
                     :or {heartbeat-ms 1000
                          on-error!
                          (fn [error] (.printStackTrace ^Throwable error))}}]
  (when-not (or (map? measurements) (fn? measurements))
    (throw (ex-info "Worker endpoint requires scheduling measurements" {})))
  (when-not (and (integer? heartbeat-ms) (pos? heartbeat-ms))
    (throw (ex-info "Heartbeat interval must be a positive integer"
                    {:heartbeat-ms heartbeat-ms})))
  (when-not (ifn? on-error!)
    (throw (ex-info "Worker endpoint error handler must be callable" {})))
  (let [holder (atom nil)
        controller (local/open-controller
                    pool worker-opts
                    {:handlers handlers
                     :send! #(send-worker-effect! @holder %)})
        executor (scheduler)
        endpoint (->WorkerEndpoint
                  controller measurements (atom -1) (long heartbeat-ms)
                  executor (atom nil) (atom nil) (atom nil) on-error!
                  (atom false))
        task (.scheduleAtFixedRate
              ^ScheduledExecutorService executor
              ^Runnable
              (reify Runnable
                (run [_]
                  (when-not @(:closed? endpoint)
                    (try
                      (publish-observation! endpoint)
                      (catch Throwable error
                        (report-error! on-error! error
                                       {:endpoint :worker
                                        :worker/id
                                        (:worker/id (local/state controller))}))))))
              (long heartbeat-ms) (long heartbeat-ms) TimeUnit/MILLISECONDS)]
    (reset! holder endpoint)
    (reset! (:heartbeat-task endpoint) task)
    endpoint))

(defn worker-state
  "Return the endpoint's immutable worker-machine state."
  [endpoint]
  (local/state (:controller endpoint)))

(defn publish-now!
  "Publish an immediate worker observation when a connection is attached."
  [endpoint]
  (publish-observation! endpoint))

(defn worker-middleware
  "Return Kabel middleware attaching `endpoint` to its router connection.

  Offers and cancellations are consumed locally; unrelated messages pass
  through for the other middleware installed on the peer."
  [endpoint]
  (fn [[supervisor peer [in out]]]
    (let [next-in (async/chan 64)
          next-out (async/chan 64)
          connection-id (UUID/randomUUID)]
      (reset! (:connection endpoint) connection-id)
      (reset! (:out endpoint) out)
      (publish-observation! endpoint)
      (async/go
        (try
          (loop []
            (when-some [message (async/<! in)]
              (if (and (wire/control-message? message)
                       (contains? worker-message-types (:type message)))
                (when-let [event (wire/worker-event
                                  (get-in (local/state (:controller endpoint))
                                          [:worker/id])
                                  message)]
                  (local/handle-event! (:controller endpoint) event))
                (async/>! next-in message))
              (recur)))
          (catch Throwable error
            (report-error! (:on-error! endpoint) error
                           {:endpoint :worker
                            :connection-id connection-id}))
          (finally
            (when (= connection-id @(:connection endpoint))
              (reset! (:out endpoint) nil)
              (reset! (:connection endpoint) nil))
            (async/close! next-in))))
      (async/go-loop []
        (when-some [message (async/<! next-out)]
          (async/>! out message)
          (recur)))
      [supervisor peer [next-in next-out]])))
