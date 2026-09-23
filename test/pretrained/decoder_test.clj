(ns pretrained.decoder-test
  (:require [clojure.test :refer [deftest is testing]]
            [pretrained.decoder :as decoder]
            [raster.dl.attention :as attn]))

(deftest decode-attention-adapts-sliding-caches-to-current-raster
  (let [q (float-array [1 2 3 4])
        k (float-array (map float (range 16)))
        v (float-array (map float (range 100 116)))
        captured (atom nil)]
    (with-redefs [attn/gqa-decode-attention-gpu!
                  (fn [actual-q actual-k actual-v out scratch cache-len
                       n-q group n-kv head-dim scale]
                    (reset! captured
                            {:q actual-q :k actual-k :v actual-v
                             :scratch-length (alength ^floats scratch)
                             :cache-len cache-len :n-q n-q :group group
                             :n-kv n-kv :head-dim head-dim :scale scale})
                    (dotimes [i (alength ^floats out)]
                      (aset ^floats out i (float (inc i)))))]
      (testing "a sliding suffix is compacted without changing its rows"
        (let [out (#'decoder/decode-attention q k v 4 2 2 2 2 0.5)]
          (is (= [1.0 2.0 3.0 4.0] (vec out)))
          (is (= [8.0 9.0 10.0 11.0 12.0 13.0 14.0 15.0]
                 (vec (:k @captured))))
          (is (= [108.0 109.0 110.0 111.0 112.0 113.0 114.0 115.0]
                 (vec (:v @captured))))
          (is (= {:scratch-length 4 :cache-len 2 :n-q 2 :group 1
                  :n-kv 2 :head-dim 2 :scale 0.5}
                 (dissoc @captured :q :k :v)))))
      (testing "a full cache remains zero-copy"
        (#'decoder/decode-attention q k v 4 0 2 2 2 0.5)
        (is (identical? k (:k @captured)))
        (is (identical? v (:v @captured)))))))
