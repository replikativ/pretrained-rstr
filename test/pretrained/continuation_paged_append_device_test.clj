(ns pretrained.continuation-paged-append-device-test
  (:require [clojure.string :as str]
            [clojure.test :refer [deftest is]]
            [pretrained.attention-state :as attention-state]
            [pretrained.continuation.page-pool :as page-pool]
            [pretrained.continuation.paged-append :as paged-append]
            [pretrained.continuation.paged-attention :as paged-attention]
            [raster.gpu.core :as gpu]))

(def ^:private level-zero-available?
  (delay
    (try
      ((requiring-resolve 'raster.gpu.ze-runtime/init!))
      true
      (catch Throwable _ false))))

(def ^:private opencl-fp16-available?
  (delay
    (try
      ((requiring-resolve 'raster.gpu.ocl-runtime/init!))
      (boolean
       (some #(str/includes? (or (:extensions %) "") "cl_khr_fp16")
             ((requiring-resolve 'raster.gpu.ocl-runtime/query-devices))))
      (catch Throwable _ false))))

(defn- half-bits
  [value]
  (Float/floatToFloat16 (float value)))

(defn- run-case
  [device-id]
  (let [session (gpu/make-session device-id)
        layout (attention-state/layout {:n-layers 1 :n-kv 1 :head-dim 4})
        key-rows (float-array [1.1 -2.2 3.3 -4.4, 5.5 -6.6 7.7 -8.8])
        value-rows (float-array [0.25 -0.5 0.75 -1.0, 1.25 -1.5 1.75 -2.0])
        sentinel (half-bits 42.0)]
    (try
      (let [pool (page-pool/open-pool!
                  session layout
                  {:page-size 2 :physical-pages 3 :dtype :half
                   :key-prefix (str "append-device-" (name device-id))})
            key-pool (get (page-pool/buffer-keys pool) [:key 0])
            value-pool (get (page-pool/buffer-keys pool) [:value 0])]
        (gpu/alloc! session
                    {:append-key-rows [:float (alength key-rows) key-rows]
                     :append-value-rows [:float (alength value-rows) value-rows]})
        (gpu/upload! session key-pool (short-array (repeat 24 sentinel)))
        (gpu/upload! session value-pool (short-array (repeat 24 sentinel)))
        (page-pool/allocate-route! pool :a 0)
        (page-pool/allocate-route! pool :b 3)
        (let [batch (paged-append/reserve-batch! pool [:a :b])]
          (is (= [4 3] (vec (paged-append/slot-values batch))))
          (with-open [runner
                      (paged-append/open-runner!
                       pool {:id [:device device-id]
                             :key-prefix (str "append-runner-" (name device-id))
                             :layer 0
                             :batch-size 2
                             :key-view (gpu/buffer-view session :append-key-rows)
                             :value-view (gpu/buffer-view session :append-value-rows)})]
            (is (identical? batch (paged-append/run! runner batch))))
          (is (= [0 3] (mapv :token-count [(page-pool/route pool :a)
                                           (page-pool/route pool :b)])))
          (paged-append/commit-batch! batch)
          (let [actual-key ^shorts (gpu/download session key-pool)
                actual-value ^shorts (gpu/download session value-pool)
                lane-by-slot {4 0, 3 1}]
            (doseq [slot (range 6)
                    component (range 4)]
              (let [index (+ (* slot 4) component)
                    lane (get lane-by-slot slot)
                    expected-key (if (some? lane)
                                   (half-bits
                                    (aget key-rows (+ (* lane 4) component)))
                                   sentinel)
                    expected-value (if (some? lane)
                                     (half-bits
                                      (aget value-rows (+ (* lane 4) component)))
                                     sentinel)]
                (is (= expected-key (aget actual-key index)))
                (is (= expected-value (aget actual-value index)))))))
        (page-pool/release-route! pool :a)
        (page-pool/release-route! pool :b)
        (page-pool/allocate-route! pool :prospective 0)
        (let [batch (paged-append/reserve-batch! pool [:prospective])]
          (with-open [append-runner
                      (paged-append/open-runner!
                       pool {:id [:ordered-append device-id]
                             :key-prefix (str "ordered-append-" (name device-id))
                             :layer 0 :batch-size 1
                             :key-view (gpu/buffer-view session :append-key-rows)
                             :value-view (gpu/buffer-view session :append-value-rows)})
                      attention-runner
                      (paged-attention/open-runner!
                       pool {:id [:ordered-attention device-id]
                             :key-prefix (str "ordered-attention-" (name device-id))
                             :layer 0 :batch-size 1 :total-query-tokens 1
                             :q-heads 1 :kv-heads 1 :qk-head-dim 4
                             :value-head-dim 4 :pages-per-sequence 1})]
            (paged-append/load-batch! append-runner batch)
            (paged-attention/load-batch!
             attention-runner
             {:continuation-ids [:prospective]
              :append-reservations (paged-append/reservation-entries batch)
              :query-values (float-array [1 0 0 0])
              :row-offsets [0 1]
              :positions [0]})
            (let [append-event (paged-append/submit! append-runner)
                  attention-event (paged-attention/submit! attention-runner)]
              (paged-append/await! append-runner append-event)
              (paged-append/commit-batch! batch)
              (let [actual ^shorts
                    (paged-attention/await! attention-runner attention-event)]
                (is (= (mapv half-bits [0.25 -0.5 0.75 -1.0])
                       (vec actual)))
                (is (= 1 (:token-count
                          (page-pool/route pool :prospective)))))))))
      (finally
        (gpu/close-session! session)))))

(deftest level-zero-resident-rows-append-into-reserved-pages
  (if-not @level-zero-available?
    (is true "Level Zero device unavailable")
    (run-case :ze:0)))

(deftest opencl-resident-rows-append-into-reserved-pages
  (if-not @opencl-fp16-available?
    (is true "OpenCL FP16 device unavailable")
    (run-case :ocl:0)))
