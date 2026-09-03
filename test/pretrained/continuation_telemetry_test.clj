(ns pretrained.continuation-telemetry-test
  (:require [clojure.test :refer [deftest is testing]]
            [pretrained.continuation.telemetry :as telemetry]))

(deftest ewma-retains-sample-confidence-and-bounds
  (let [calibration (-> {}
                        (telemetry/record :latency-ms 10.0 0.25)
                        (telemetry/record :latency-ms 14.0 0.25))
        observation (:latency-ms calibration)]
    (is (= 2 (:count observation)))
    (is (= 11.0 (:ewma observation)))
    (is (= 10.0 (:minimum observation)))
    (is (= 14.0 (:maximum observation) (:last observation)))
    (is (= 11.0 (telemetry/estimate calibration :latency-ms)))))

(deftest telemetry-rejects-invalid-observations
  (testing "negative and non-finite samples"
    (is (thrown? clojure.lang.ExceptionInfo
                 (telemetry/observation -1)))
    (is (thrown? clojure.lang.ExceptionInfo
                 (telemetry/observation Double/NaN))))
  (testing "invalid smoothing"
    (is (thrown? clojure.lang.ExceptionInfo
                 (telemetry/update-observation nil 1.0 0.0)))))
