(ns pretrained.continuation.controller.discovery
  "Pure latest-observation registry for cluster candidate discovery."
  (:require [pretrained.continuation.controller.candidates :as candidates]))

(defn initial-state
  "Return an empty worker-observation registry."
  []
  {:discovery/workers {}})

(defn observe
  "Accept `observation` when its `[epoch sequence]` is newer for the worker.

  Equal versions are idempotent. An older process epoch or heartbeat sequence
  is ignored, preventing delayed Kabel frames from replacing current capacity.
  Returns the updated immutable registry."
  [state observation]
  (let [state (or state (initial-state))
        observation (candidates/worker-observation observation)
        worker-id (:worker/id observation)
        current (get-in state [:discovery/workers worker-id])
        version (juxt :worker/epoch :worker/sequence)]
    (if (and current
             (not (neg? (compare (version current) (version observation)))))
      state
      (assoc-in state [:discovery/workers worker-id] observation))))

(defn remove-worker
  "Remove `worker-id`, for example after an independently established timeout."
  [state worker-id]
  (update state :discovery/workers dissoc worker-id))

(defn observations
  "Return observations in stable worker-id order."
  [state]
  (->> (:discovery/workers state)
       vals
       (sort-by (comp str :worker/id))
       vec))
