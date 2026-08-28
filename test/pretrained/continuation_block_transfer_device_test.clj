(ns pretrained.continuation-block-transfer-device-test
  (:require [clojure.string :as str]
            [clojure.test :refer [deftest is]]
            [pretrained.attention-state :as attention-state]
            [pretrained.continuation :as continuation]
            [pretrained.continuation.page-pool :as page-pool]
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

(defn- direction-commands
  [pool direction]
  (reduce + 0
          (for [[[measured-direction _ _] counters]
                (:counters (page-pool/transfer-stats pool))
                :when (= direction measured-direction)]
            (:commands counters))))

(defn- run-case
  [device-id]
  (let [session (gpu/make-session device-id)
        model {:n-layers 1 :n-kv 1 :head-dim 2}
        layout (attention-state/layout model)
        pool (page-pool/open-pool!
              session layout {:page-size 2 :physical-pages 12 :dtype :half
                              :key-prefix (str "block-device-" (name device-id))})
        blocker-ids (mapv #(keyword (str "blocker-" %)) (range 12))
        token-count 12
        payload (short-array
                 (map #(Float/floatToFloat16 (float (- % 24)))
                      (range (* 2 token-count 2))))
        descriptor {:chunk/start 0
                    :chunk/token-count token-count
                    :chunk/layout
                    (assoc-in (continuation/model-layout model)
                              [:attention-state :dtype] :float16)}]
    (try
      (doseq [continuation-id blocker-ids]
        (page-pool/allocate-route! pool continuation-id 2))
      (doseq [continuation-id (take-nth 2 blocker-ids)]
        (page-pool/release-route! pool continuation-id))
      (is (= [0 2 4 6 8 10]
             (:pages (page-pool/allocate-route! pool :fragmented token-count))))
      (page-pool/restore-chunk! pool :fragmented descriptor payload)
      (let [restored (:chunk/payload
                      (page-pool/export-chunk
                       pool :fragmented "block-device-fixture" descriptor))]
        (is (= (vec payload) (vec restored))))
      (is (= 4 (direction-commands pool :upload))
          "restore uploads indices plus two slabs; capture uploads indices")
      (is (= 2 (direction-commands pool :download))
          "capture downloads two dense slab buffers")
      (finally
        (page-pool/close-transfer-engines! pool)
        (gpu/close-session! session)))))

(deftest level-zero-fragmented-pages-roundtrip-through-block-staging
  (if-not @level-zero-available?
    (is true "Level Zero device unavailable")
    (run-case :ze:0)))

(deftest opencl-fragmented-pages-roundtrip-through-block-staging
  (if-not @opencl-fp16-available?
    (is true "OpenCL FP16 device unavailable")
    (run-case :ocl:0)))
