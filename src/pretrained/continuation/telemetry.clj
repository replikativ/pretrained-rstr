(ns pretrained.continuation.telemetry
  "Small immutable EWMA statistics for live worker calibration.")

(def default-alpha
  "Default weight assigned to the newest live measurement."
  0.2)

(defn observation
  "Return a normalized first observation for non-negative finite `value`."
  [value]
  (let [value (double value)]
    (when-not (and (Double/isFinite value) (not (neg? value)))
      (throw (ex-info "Telemetry observations must be non-negative and finite"
                      {:value value})))
    {:count 1
     :ewma value
     :minimum value
     :maximum value
     :last value}))

(defn update-observation
  "Update an immutable EWMA observation with `value` and smoothing `alpha`.

  `alpha` must be in `(0,1]`. The result retains exact count/min/max/last values
  alongside the exponentially weighted estimate used by worker scheduling."
  ([current value]
   (update-observation current value default-alpha))
  ([current value alpha]
   (let [alpha (double alpha)]
     (when-not (and (Double/isFinite alpha) (< 0.0 alpha) (<= alpha 1.0))
       (throw (ex-info "Telemetry EWMA alpha must be in (0,1]"
                       {:alpha alpha})))
     (let [sample (observation value)]
       (if-not current
         sample
         (let [value (:last sample)
               previous (double (:ewma current))]
           {:count (inc (long (:count current)))
            :ewma (+ (* alpha value) (* (- 1.0 alpha) previous))
            :minimum (min (double (:minimum current)) value)
            :maximum (max (double (:maximum current)) value)
            :last value}))))))

(defn record
  "Record one named `value` in a calibration map and return the updated map."
  ([calibration metric value]
   (record calibration metric value default-alpha))
  ([calibration metric value alpha]
   (when-not (keyword? metric)
     (throw (ex-info "Telemetry metric names must be keywords"
                     {:metric metric})))
   (update (or calibration {}) metric update-observation value alpha)))

(defn estimate
  "Return the current EWMA for `metric`, or nil when it has no samples."
  [calibration metric]
  (get-in calibration [metric :ewma]))
