(ns pretrained.real-openai-cluster-demo-test
  (:require [clojure.test :refer [deftest is]]
            [pretrained.real-openai-cluster-demo :as demo]))

(def ^:private gib (* 1024 1024 1024))

(deftest resource-admission-is-explicit-and-overrideable
  (let [quiet {:available-bytes (* 20 gib)
               :swap-total-bytes (* 8 gib)
               :swap-free-bytes (* 4 gib)
               :load-one 2.0
               :processors 8}
        pressured (assoc quiet
                         :available-bytes (* 3 gib)
                         :swap-free-bytes (* 64 1024 1024)
                         :load-one 16.0)]
    (is (:admitted? (demo/resource-admission quiet)))
    (is (= [:insufficient-available-memory
            :insufficient-free-swap
            :host-load-too-high]
           (:reasons (demo/resource-admission pressured))))
    (is (:admitted?
         (demo/resource-admission
          pressured
          {:minimum-available-bytes (* 2 gib)
           :minimum-swap-free-bytes (* 32 1024 1024)
           :maximum-load-per-processor 3.0})))))
