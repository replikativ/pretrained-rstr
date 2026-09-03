(ns pretrained.continuation-calibration-test
  (:require [clojure.test :refer [deftest is]]
            [pretrained.continuation.calibration :as calibration]
            [pretrained.continuation.manager :as manager]
            [pretrained.continuation.page-pool :as page-pool]
            [pretrained.continuation.paged-runtime :as paged-runtime]))

(deftest live-samples-override-configured-worker-fallbacks
  (with-redefs [paged-runtime/state
                (fn [_]
                  {:calibration
                   {:prefill-ms-per-token {:count 4 :ewma 0.5}
                    :first-token-ms {:count 3 :ewma 12.0}}})
                manager/stats
                (fn [_]
                  {:calibration {:checkpoint-ms {:count 2 :ewma 80.0}}})
                page-pool/transfer-stats
                (fn [_]
                  {:calibration
                   {:gpu-upload-bytes-per-ms {:count 5 :ewma 2000.0}}})]
    (let [result
          (calibration/worker-measurements
           ::runtime ::cache ::pool
           {:worker/node "worker-a"
            :worker/prefill-ms-per-token 9.0
            :worker/first-token-ms 99.0
            :worker/gpu-restore-bytes-per-ms 100.0})]
      (is (= 0.5 (:worker/prefill-ms-per-token result)))
      (is (= 12.0 (:worker/first-token-ms result)))
      (is (= 2000.0 (:worker/gpu-restore-bytes-per-ms result)))
      (is (= 80.0 (get-in result
                          [:worker/live-calibration
                           :checkpoint :checkpoint-ms :ewma]))))))

(deftest configured-values-survive-a-cold-start
  (with-redefs [paged-runtime/state (constantly {})
                manager/stats (constantly {})
                page-pool/transfer-stats (constantly {})]
    (let [base {:worker/prefill-ms-per-token 9.0
                :worker/first-token-ms 99.0
                :worker/gpu-restore-bytes-per-ms 100.0}
          measurements (calibration/measurements-fn
                        ::runtime ::cache ::pool base)
          result (measurements)]
      (is (= base (select-keys result (keys base))))
      (is (= {:runtime nil :checkpoint nil :transfer nil}
             (:worker/live-calibration result))))))
