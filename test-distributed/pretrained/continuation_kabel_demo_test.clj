(ns pretrained.continuation-kabel-demo-test
  (:require [clojure.test :refer [deftest is]]
            [pretrained.continuation-kabel-demo :as demo]))

(deftest two-worker-openai-stream-routes-over-live-kabel-sockets
  (let [result (demo/run-openai-live-simulation)]
    (is (= 200 (:http-status result)))
    (is (= "text/event-stream; charset=utf-8" (:content-type result)))
    (is (= :fast-gpu (:selected-worker result)))
    (is (= :completed (:phase result)))
    (is (= "<101><102>" (:text result)))
    (is (zero? (:cached-token-count result)))
    (is (= 2 (:observed-workers result)))))
