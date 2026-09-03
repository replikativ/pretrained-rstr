(ns pretrained.continuation.calibration
  "Compose live runtime, cache-manager, and transfer EWMAs for worker routing."
  (:require [pretrained.continuation.manager :as manager]
            [pretrained.continuation.page-pool :as page-pool]
            [pretrained.continuation.paged-runtime :as paged-runtime]
            [pretrained.continuation.telemetry :as telemetry]))

(defn snapshot
  "Return immutable live calibration from `runtime`, `cache`, and `pool`.

  The nested observations retain counts and bounds so operators can distinguish
  a stable estimate from a first sample. `:worker-overrides` contains only
  estimates that currently have samples and can safely override configured
  worker measurement defaults."
  [runtime cache pool]
  (let [runtime-calibration (:calibration (paged-runtime/state runtime))
        manager-calibration (:calibration (manager/stats cache))
        transfer-calibration (:calibration (page-pool/transfer-stats pool))
        prefill (telemetry/estimate runtime-calibration :prefill-ms-per-token)
        first-token (telemetry/estimate runtime-calibration :first-token-ms)
        gpu-upload (telemetry/estimate transfer-calibration
                                       :gpu-upload-bytes-per-ms)]
    {:runtime runtime-calibration
     :checkpoint manager-calibration
     :transfer transfer-calibration
     :worker-overrides
     (cond-> {}
       prefill (assoc :worker/prefill-ms-per-token prefill)
       first-token (assoc :worker/first-token-ms first-token)
       gpu-upload (assoc :worker/gpu-restore-bytes-per-ms gpu-upload))}))

(defn worker-measurements
  "Merge sampled routing costs into configured worker `base` measurements.

  Configuration remains the fallback until a metric has at least one live
  sample. The complete calibration is attached as `:worker/live-calibration`
  for policy functions and observability; the ordinary candidate protocol
  ignores this additional field."
  [runtime cache pool base]
  (let [{:keys [worker-overrides] :as live} (snapshot runtime cache pool)]
    (assoc (merge base worker-overrides)
           :worker/live-calibration (dissoc live :worker-overrides))))

(defn measurements-fn
  "Return a zero-argument live measurement supplier for a Kabel worker endpoint."
  [runtime cache pool base]
  #(worker-measurements runtime cache pool base))
