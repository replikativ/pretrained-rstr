(ns pretrained.laya-cpu-benchmark-test
  (:require [clojure.test :refer [deftest is testing]]
            [pretrained.laya-cpu-benchmark :as benchmark]))

(deftest shared-benchmark-fixture-is-stable
  (let [{:keys [id state questions]} (benchmark/fixture)]
    (is (= "typed-decisions-four-question-v1" id))
    (is (= 4 (count questions)))
    (is (= #{"choice" "score" "noul"}
           (set (map :type (vals questions)))))
    (is (= "Duplicate charge on invoice #4411" (:subject state)))))

(deftest timing-statistics-are-explicit
  (testing "median and p90 are drawn from sorted samples"
    (is (= {:samples-ms [5.0 1.0 3.0 2.0 4.0]
           :min-ms 1.0 :median-ms 3.0 :p90-ms 5.0 :max-ms 5.0}
           (benchmark/statistics [5 1 3 2 4]))))
  (testing "even sample counts use the same nearest-sample rule as Python"
    (is (= {:samples-ms [4.0 1.0 3.0 2.0]
            :min-ms 1.0 :median-ms 3.0 :p90-ms 4.0 :max-ms 4.0}
           (benchmark/statistics [4 1 3 2])))))
