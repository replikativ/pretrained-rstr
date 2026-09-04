(ns pretrained.openai.cluster
  "Lifecycle adapter joining OpenAI HTTP ingress to a Kabel continuation router.

  The router remains transport-neutral and owns candidate selection/fencing;
  the HTTP server owns JSON, SSE, socket backpressure, and disconnects. Worker
  connections attach through `router-middleware`, so tensor payloads continue
  to use their Datahike/Konserve placement path rather than HTTP or Kabel."
  (:require [pretrained.continuation.controller.kabel :as kabel]
            [pretrained.openai.server :as openai-server])
  (:import (java.io Closeable)))

(defrecord ClusterOpenAIServer [http router closed?]
  Closeable
  (close [_]
    (when (compare-and-set! closed? false true)
      ;; HTTP close cancels in-flight requests while the router is still able
      ;; to fence and forward those cancellations to workers.
      (try
        (.close ^Closeable http)
        (finally
          (.close ^Closeable router))))))

(defn open-server
  "Open one OpenAI HTTP server backed by a Kabel continuation router.

  `database` is the Datahike catalog connection or value used for exact-prefix
  candidate planning. HTTP options are the same callback-free options accepted
  by `pretrained.openai.server/open-server`: `:models`, `:tokenize-chat`,
  `:decode-token`, `:decode-tokens`, `:max-pending-events`, and
  `:server-options`. Router timing and error options may be supplied under
  `:router-options`.

  Attach worker WebSocket connections with `(router-middleware result)`. Close
  the returned value to stop ingress first and then the router."
  [database {:keys [models tokenize-chat decode-token decode-tokens
                    max-pending-events server-options router-options]}]
  (let [holder (atom nil)
        router
        (kabel/open-router-endpoint
         database
         (assoc (or router-options {})
                :deliver!
                (fn [delivery]
                  (when-let [http @holder]
                    (openai-server/deliver! http delivery)))))]
    (try
      (let [http
            (openai-server/open-server
             (cond-> {:models models
                      :tokenize-chat tokenize-chat
                      :decode-token decode-token
                      :decode-tokens decode-tokens
                      :submit! #(kabel/submit! router %)
                      :cancel! #(kabel/cancel-request! router %)
                      :server-options (or server-options {})}
               max-pending-events
               (assoc :max-pending-events max-pending-events)))
            result (->ClusterOpenAIServer http router (atom false))]
        (reset! holder http)
        result)
      (catch Throwable error
        (.close ^Closeable router)
        (throw error)))))

(defn router-middleware
  "Return the Kabel middleware to attach worker connections to this server."
  [server]
  (kabel/router-middleware (:router server)))

(defn observations
  "Return the router's current worker observations."
  [server]
  (kabel/observations (:router server)))

(defn router-state
  "Return the immutable cluster-router state."
  [server]
  (kabel/router-state (:router server)))

(defn local-port
  "Return the OpenAI HTTP listener's bound port."
  [server]
  (openai-server/local-port (:http server)))
