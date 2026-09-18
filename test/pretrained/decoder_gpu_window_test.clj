(ns pretrained.decoder-gpu-window-test
  (:require [clojure.test :refer [deftest is]]
            [pretrained.decoder-gpu :as decoder-gpu]))

(deftest contiguous-decode-refuses-spans-beyond-the-window
  (let [guard #'decoder-gpu/require-contiguous-window!
        gemma {:model {:n-layers 6
                       :desc {:flags {:global-layer-pattern 6
                                      :sliding-window {:size 512}}}}}]
    (is (nil? (guard gemma 512)) "a span equal to the window is exact")
    (is (thrown-with-msg? clojure.lang.ExceptionInfo #"use :cache-mode :paged"
                          (guard gemma 513)))
    (is (nil? (guard {:model {:n-layers 2 :desc {:flags {}}}} 100000))
        "models without windows are unaffected")))
