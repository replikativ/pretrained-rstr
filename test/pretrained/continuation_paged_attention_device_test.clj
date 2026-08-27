(ns pretrained.continuation-paged-attention-device-test
  (:require [clojure.test :refer [deftest is]]
            [pretrained.attention-state :as attention-state]
            [pretrained.continuation.page-pool :as page-pool]
            [pretrained.continuation.paged-attention :as paged-attention]
            [raster.compiler.ir.attention :as attention]
            [raster.gpu.core :as gpu]))

(def ^:private level-zero-available?
  (delay
    (try
      ((requiring-resolve 'raster.gpu.ze-runtime/init!))
      true
      (catch Throwable _
        false))))

(defn- decode-halfs
  [^shorts values]
  (mapv #(double (Float/float16ToFloat %)) values))

(deftest resident-pages-feed-a-real-raster-attention-graph
  (if-not @level-zero-available?
    (is true "Level Zero device unavailable")
    (let [session (gpu/make-session :ze:0)
          model {:n-layers 1 :n-kv 1 :head-dim 2}]
      (try
        (let [pool (page-pool/open-pool!
                    session (attention-state/layout model)
                    {:page-size 2 :physical-pages 4 :dtype :half
                     :key-prefix "device-page"})]
          (page-pool/allocate-route! pool :a 0)
          (page-pool/append-token!
           pool :a {[:key 0] (float-array [1 0])
                    [:value 0] (float-array [10 20])})
          (page-pool/append-token!
           pool :a {[:key 0] (float-array [0 1])
                    [:value 0] (float-array [30 40])})
          (page-pool/allocate-route! pool :b 0)
          (page-pool/append-token!
           pool :b {[:key 0] (float-array [1 1])
                    [:value 0] (float-array [5 7])})
          (with-open [runner
                      (paged-attention/open-runner!
                       pool {:id :device-fixture
                             :key-prefix "device-attention"
                             :layer 0
                             :batch-size 2
                             :total-query-tokens 2
                             :q-heads 1
                             :kv-heads 1
                             :qk-head-dim 2
                             :value-head-dim 2
                             :pages-per-sequence 2})]
            (let [actual
                  (decode-halfs
                   (paged-attention/run!
                    runner
                    {:continuation-ids [:a :b]
                     :query-values (float-array [1 0, 1 0])
                     :row-offsets [0 1 2]
                     :positions [1 0]}))
                  expected [16.6048 26.6048 5.0 7.0]]
              (is (= 4 (count actual)))
              (is (every? true?
                          (map #(< (Math/abs (- %1 %2)) 0.04)
                               expected actual)))))
          (with-open [runner
                      (paged-attention/open-runner!
                       pool {:id :windowed-device-fixture
                             :key-prefix "windowed-device-attention"
                             :layer 0
                             :batch-size 2
                             :total-query-tokens 2
                             :q-heads 1
                             :kv-heads 1
                             :qk-head-dim 2
                             :value-head-dim 2
                             :pages-per-sequence 2
                             :visibility (attention/visibility
                                          {:causal? true :window-left 0})})]
            (let [actual
                  (decode-halfs
                   (paged-attention/run!
                    runner
                    {:continuation-ids [:a :b]
                     :query-values (float-array [1 0, 1 0])
                     :row-offsets [0 1 2]
                     :positions [1 0]}))]
              (is (= [30.0 40.0 5.0 7.0] actual))))
          (gpu/alloc! session
                      {:resident-query [:float 4 (float-array [1 0 1 0])]
                       :resident-output [:float 4 nil]})
          (let [query-view (gpu/buffer-view session :resident-query)
                output-view (gpu/buffer-view session :resident-output)]
            (with-open [runner
                        (paged-attention/open-runner!
                         pool {:id :resident-device-fixture
                               :key-prefix "resident-device-attention"
                               :layer 0
                               :batch-size 2
                               :total-query-tokens 2
                               :q-heads 1
                               :kv-heads 1
                               :qk-head-dim 2
                               :value-head-dim 2
                               :pages-per-sequence 2
                               :query-dtype :float
                               :output-dtype :float
                               :query-view query-view
                               :output-view output-view})]
              (is (identical?
                   output-view
                   (paged-attention/run!
                    runner
                    {:continuation-ids [:a :b]
                     :row-offsets [0 1 2]
                     :positions [1 0]}))))
            (let [actual (mapv double ^floats (gpu/download session :resident-output))
                  expected [16.6048 26.6048 5.0 7.0]]
              (is (every? true?
                          (map #(< (Math/abs (- %1 %2)) 1.0e-4)
                               expected actual))))
            (is (some? (gpu/buffer session :resident-query)))
            (is (some? (gpu/buffer session :resident-output)))))
        (finally
          (gpu/close-session! session))))))
