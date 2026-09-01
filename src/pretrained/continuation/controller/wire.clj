(ns pretrained.continuation.controller.wire
  "EDN wire values for continuation control messages.

  The maps are directly suitable for Kabel's `:type`-keyed bus. Only
  cluster-control effects are encoded here; timers, consumer delivery, GPU
  operations, and tensor payloads remain local to their owning interpreter.")

(def ^:private wire-types
  #{:continuation/offer
    :continuation/cancel
    :continuation/offer-result
    :continuation/result
    :continuation/worker-observation
    :continuation/worker-unavailable})

(defn effect->message
  "Encode one router/worker network effect as a Kabel-friendly EDN map.

  Throws when `effect` is local-only. This explicit failure prevents GPU
  operation values or timer bookkeeping from accidentally entering the wire
  protocol."
  [effect]
  (case (:effect/op effect)
    :router/send-offer
    (-> effect
        (dissoc :effect/op :effect/to)
        (assoc :type :continuation/offer
               :message/to (:effect/to effect)))

    :router/send-cancel
    (-> effect
        (dissoc :effect/op :effect/to)
        (assoc :type :continuation/cancel
               :message/to (:effect/to effect)))

    :worker/send-offer-result
    (-> effect
        (dissoc :effect/op :effect/to)
        (assoc :type :continuation/offer-result
               :message/to :router))

    :worker/send-result
    (-> effect
        (dissoc :effect/op :effect/to)
        (assoc :type :continuation/result
               :message/to :router))

    (throw (ex-info "Continuation effect is local and cannot be encoded"
                    {:effect/op (:effect/op effect)}))))

(defn worker-event
  "Decode an offer or cancellation addressed to `worker-id`.

  Returns nil for another destination. Throws for a message type that does not
  belong at a worker."
  [worker-id message]
  (if (not= worker-id (:message/to message))
    nil
    (case (:type message)
      :continuation/offer
      (-> message
          (dissoc :type :message/to)
          (assoc :event/type :assignment/offered))

      :continuation/cancel
      (-> message
          (dissoc :type :message/to)
          (assoc :event/type :assignment/cancelled))

      (throw (ex-info "Continuation message is not worker-addressed"
                      {:type (:type message) :worker/id worker-id})))))

(defn router-event
  "Decode one worker-to-router control message.

  Returns nil for a non-router destination. Throws for an unsupported message
  type."
  [message]
  (when (= :router (:message/to message))
    (case (:type message)
      :continuation/offer-result
      (-> message
          (dissoc :type :message/to)
          (assoc :event/type :worker/offer-result))

      :continuation/result
      (-> message
          (dissoc :type :message/to)
          (assoc :event/type :worker/result))

      :continuation/worker-unavailable
      (-> message
          (dissoc :type :message/to)
          (assoc :event/type :worker/unavailable))

      (throw (ex-info "Continuation message is not router-addressed"
                      {:type (:type message)})))))

(defn observation->message
  "Encode one worker observation for the router's discovery registry."
  [observation]
  {:type :continuation/worker-observation
   :message/to :router
   :message/observation observation})

(defn message->observation
  "Return a router-addressed worker observation, or nil for another message."
  [message]
  (when (and (= :continuation/worker-observation (:type message))
             (= :router (:message/to message)))
    (:message/observation message)))

(defn control-message?
  "Return true when `value` has a recognized continuation wire type."
  [value]
  (and (map? value) (contains? wire-types (:type value))))
