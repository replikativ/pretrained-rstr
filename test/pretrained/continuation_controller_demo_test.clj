(ns pretrained.continuation-controller-demo-test
  (:require [clojure.test :refer [deftest is]]
            [pretrained.continuation-controller-demo :as demo]))

(deftest database-facts-and-worker-observations-drive-the-simulation
  (let [result (demo/run-database-simulation)]
    (is (= :warm-ssd (:selected-worker result)))
    (is (= :warm-ssd
           (:candidate/worker-id (first (:candidates result)))))
    (is (= :ssd
           (:candidate/cache-tier (first (:candidates result)))))
    (is (some #(and (= :busy-gpu (:candidate/worker-id %))
                    (= :gpu (:candidate/cache-tier %)))
              (:candidates result)))
    (is (= :completed (get-in result [:response :response/value :status])))))
