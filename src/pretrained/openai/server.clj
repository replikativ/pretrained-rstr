(ns pretrained.openai.server
  "Optional HTTP-kit transport for the OpenAI-compatible chat boundary.

  This namespace requires the `:openai-server` alias. It uses the replikativ
  http-kit fork's per-connection writability signal in addition to a bounded
  application event queue. Controller delivery therefore never waits for a
  slow socket and cannot grow an unbounded backlog."
  (:require [clojure.data.json :as json]
            [clojure.java.io :as io]
            [org.httpkit.server :as http]
            [pretrained.openai :as openai])
  (:import (java.io Closeable InputStream Reader)
           (java.util.concurrent ExecutorService Executors)))

(def ^:private json-headers
  {"content-type" "application/json; charset=utf-8"})

(def ^:private stream-headers
  {"content-type" "text/event-stream; charset=utf-8"
   "cache-control" "no-cache"
   "connection" "keep-alive"})

(defrecord OpenAIServer
    [stop! sessions submit! cancel! models request-options sender
     max-pending-events closed?]
  Closeable
  (close [this]
    (when (compare-and-set! closed? false true)
      (doseq [[request-id {:keys [channel]}] @sessions]
        (try (cancel! request-id) (catch Throwable _))
        (http/close channel))
      (reset! sessions {})
      (when stop! (stop!))
      (.shutdownNow ^ExecutorService sender))))

(declare deliver!)

(defn- json-response
  [status body]
  {:status status
   :headers json-headers
   :body (json/write-str body)})

(defn- request-body
  [request]
  (let [body (:body request)]
    (cond
      (string? body) (json/read-str body :key-fn keyword)
      (instance? Reader body) (json/read body :key-fn keyword)
      (instance? InputStream body)
      (with-open [reader (io/reader body)]
        (json/read reader :key-fn keyword))
      :else (throw (ex-info "Request body is not readable JSON"
                            {:openai/error
                             {:message "Request body must contain JSON"
                              :type "invalid_request_error"
                              :param nil :code nil}})))))

(defn- pop-event!
  [session]
  (let [value (volatile! nil)]
    (swap! (:pending session)
           (fn [queue]
             (if (seq queue)
               (do (vreset! value (peek queue)) (pop queue))
               queue)))
    @value))

(declare schedule-drain!)

(defn- finish-session!
  [server request-id session]
  (swap! (:sessions server)
         (fn [sessions]
           (if (identical? session (get sessions request-id))
             (dissoc sessions request-id)
             sessions))))

(defn- drain!
  [server request-id session]
  (let [channel (:channel session)]
    (loop []
      (cond
        (or @(:closed? server) (not (http/open? channel)))
        (do
          (reset! (:draining? session) false)
          (finish-session! server request-id session))

        (not (http/writable? channel))
        (do
          (reset! (:draining? session) false)
          (http/on-writable channel #(schedule-drain! server request-id session)))

        :else
        (if-let [{:keys [payload close?]} (pop-event! session)]
          (if (http/send! channel payload close?)
            (if close?
              (do
                (reset! (:draining? session) false)
                (finish-session! server request-id session))
              (recur))
            (do
              (reset! (:draining? session) false)
              (finish-session! server request-id session)))
          (do
            (reset! (:draining? session) false)
            ;; An enqueue can race the empty observation. Re-acquire the drain
            ;; flag if work appeared after it was released.
            (when (seq @(:pending session))
              (schedule-drain! server request-id session))))))))

(defn- schedule-drain!
  [server request-id session]
  (when (and (not @(:closed? server))
             (compare-and-set! (:draining? session) false true))
    (.execute ^ExecutorService (:sender server)
              ^Runnable #(drain! server request-id session))))

(defn- overflow!
  [server request-id session]
  (when (compare-and-set! (:overflowed? session) false true)
    (finish-session! server request-id session)
    (try ((:cancel! server) request-id) (catch Throwable _))
    (http/close (:channel session))))

(defn- enqueue!
  [server request-id session event]
  (let [event (if (and (:stream? session)
                       (compare-and-set! (:headers-sent? session) false true))
                (update event :payload
                        (fn [body]
                          {:status 200 :headers stream-headers :body body}))
                event)
        accepted? (volatile! false)]
    (swap! (:pending session)
           (fn [queue]
             (if (< (count queue) (:max-pending-events server))
               (do (vreset! accepted? true) (conj queue event))
               queue)))
    (if @accepted?
      (schedule-drain! server request-id session)
      (overflow! server request-id session))))

(defn deliver!
  "Deliver one fenced cluster-controller response to its HTTP session.

  Unknown or already closed request ids are ignored. This function is safe for
  controller/Kabel threads: it only performs bounded queue mutation and
  schedules nonblocking socket writes on the ingress sender executor."
  [server delivery]
  (let [request-id (:request/id delivery)]
    (when-let [session (get @(:sessions server) request-id)]
      (when (contains? #{:completed :cancelled :error}
                       (:response/type delivery))
        ;; HTTP-kit may invoke on-close synchronously from a close-after-send.
        ;; Mark terminal delivery first so that normal completion is not
        ;; misclassified as a client disconnect and redundantly cancelled.
        (reset! (:terminal? session) true))
      (if (:stream? session)
        (doseq [value (openai/stream-values (:context session) delivery)]
          (enqueue! server request-id session
                    {:payload (openai/sse value) :close? (= :done value)}))
        (when (contains? #{:completed :cancelled :error}
                         (:response/type delivery))
          (enqueue! server request-id session
                    {:payload (json-response
                               (if (= :completed (:response/type delivery)) 200 500)
                               (openai/completion-response
                                (:context session) delivery))
                     :close? true}))))))

(defn- model-response
  [models]
  {:object "list"
   :data (mapv (fn [id]
                 {:id id :object "model" :created 0 :owned_by "replikativ"})
               (sort (keys models)))})

(defn- open-chat!
  [server request]
  (try
    (let [{context :openai/context generation :generation/request}
          (openai/normalize-chat-request
           (request-body request) (:request-options server))
          request-id (:request/id generation)
          context (assoc context
                         :decode-token
                         (get-in server [:request-options :decode-token])
                         :decode-tokens
                         (get-in server [:request-options :decode-tokens]))]
      (http/as-channel
       request
       {:on-open
        (fn [channel]
          (let [session {:channel channel
                         :context context
                         :stream? (:stream? context)
                         :pending (atom clojure.lang.PersistentQueue/EMPTY)
                         :draining? (atom false)
                         :headers-sent? (atom false)
                         :terminal? (atom false)
                         :overflowed? (atom false)}]
            (swap! (:sessions server) assoc request-id session)
            (try
              ((:submit! server) generation)
              (catch Throwable error
                (deliver! server {:request/id request-id
                                  :response/type :error
                                  :response/error error})))))
        :on-close
        (fn [_ _]
          (when-let [session (get @(:sessions server) request-id)]
            (finish-session! server request-id session)
            (when-not @(:terminal? session)
              (try ((:cancel! server) request-id) (catch Throwable _)))))}))
    (catch Throwable error
      (json-response 400 (openai/error-object error)))))

(defn handler
  "Return the Ring handler owned by `server`."
  [server]
  (fn [request]
    (let [method (:request-method request)
          uri (:uri request)]
      (cond
        (and (= :get method) (= "/v1/models" uri))
        (json-response 200 (model-response (:models server)))

        (and (= :post method) (= "/v1/chat/completions" uri))
        (open-chat! server request)

        :else
        (json-response 404
                       {:error {:message "Not found" :type "invalid_request_error"
                                :param nil :code nil}})))))

(defn open-server
  "Start an optional OpenAI-compatible HTTP server.

  Required options are `:models`, `:tokenize-chat`, `:decode-token`,
  `:decode-tokens`, `:submit!`, and `:cancel!`. The submit callback receives one
  validated continuation generation request. Configure the paired cluster
  router with `(partial deliver! server)`.

  `:max-pending-events` defaults to 256. HTTP-kit defaults use a 64 KiB high
  watermark, 32 KiB low watermark, and 1 MiB hard per-connection queued-byte
  ceiling. All ordinary `run-server` options may be supplied under
  `:server-options`; `:port` defaults to 8080."
  [{:keys [models tokenize-chat decode-token decode-tokens submit! cancel!
           max-pending-events server-options]
    :or {max-pending-events 256 server-options {}}}]
  (when-not (and (map? models) (seq models)
                 (every? string? (keys models))
                 (every? string? (vals models)))
    (throw (ex-info "OpenAI server requires public model ids and fingerprints" {})))
  (when-not (every? ifn? [tokenize-chat decode-token decode-tokens submit! cancel!])
    (throw (ex-info "OpenAI server callbacks must be callable" {})))
  (when-not (and (integer? max-pending-events) (pos? max-pending-events))
    (throw (ex-info "OpenAI server pending-event bound must be positive"
                    {:max-pending-events max-pending-events})))
  (let [holder (atom nil)
        server (map->OpenAIServer
                {:sessions (atom {})
                 :submit! submit!
                 :cancel! cancel!
                 :models models
                 :request-options {:models models
                                   :tokenize-chat tokenize-chat
                                   :decode-token decode-token
                                   :decode-tokens decode-tokens}
                 :sender (Executors/newSingleThreadExecutor)
                 :max-pending-events (long max-pending-events)
                 :closed? (atom false)})
        options (merge {:port 8080
                        :queue-high-water-bytes (* 64 1024)
                        :queue-low-water-bytes (* 32 1024)
                        :max-queued-bytes (* 1024 1024)}
                       server-options)
        stop! (http/run-server (fn [request] ((handler @holder) request)) options)
        server (assoc server :stop! stop!)]
    (reset! holder server)
    server))

(defn local-port
  "Return the bound TCP port, including when configured with port zero."
  [server]
  (:local-port (meta (:stop! server))))
