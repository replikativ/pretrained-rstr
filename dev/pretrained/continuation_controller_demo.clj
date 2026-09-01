(ns pretrained.continuation-controller-demo
  "Model-free deterministic cluster-continuation demonstration."
  (:require [datahike.api :as d]
            [pretrained.continuation.catalog :as catalog]
            [pretrained.continuation.chunk :as chunk]
            [pretrained.continuation.controller.candidates :as candidates]
            [pretrained.continuation.controller.sim :as sim]
            [pretrained.continuation.placement :as placement]))

(defn- candidate
  [worker-id tier cached-token-count costs]
  (merge
   {:candidate/worker-id worker-id
    :candidate/worker-epoch 0
    :candidate/cache-tier tier
    :candidate/cached-token-count cached-token-count
    :candidate/cached-bytes (* cached-token-count 1024)
    :candidate/page-size 16
    :candidate/free-pages 128
    :candidate/evictable-pages 0
    :candidate/max-context 8192
    :candidate/online? true
    :candidate/model-loaded? true}
   costs))

(defn run-simulation
  "Run a deterministic two-worker continuation request.

  The worker with the longest prefix is intentionally busy. The router chooses
  the lower predicted TTFT, both pure state machines complete, and the returned
  trace can be inspected in a REPL. No model, GPU, network, or object store is
  required."
  []
  (let [model "gemma-fixture-v1"
        request {:request/id :demo/request
                 :request/model-fingerprint model
                 :request/tokens (vec (range 257))
                 :request/max-new-tokens 8
                 :request/simulated-output [42 43 44 45 46 47 48 49]}
        workers
        {:busy-gpu {:worker/epoch 0 :worker/models #{model}
                    :worker/free-pages 128 :worker/evictable-pages 0}
         :warm-ssd {:worker/epoch 0 :worker/models #{model}
                    :worker/free-pages 128 :worker/evictable-pages 0}}
        candidates
        [(candidate :busy-gpu :gpu 256
                    {:candidate/queue-ms 45
                     :candidate/prefix-load-ms 0
                     :candidate/gpu-restore-ms 0
                     :candidate/prefill-ms-per-token 0.2
                     :candidate/first-token-ms 2})
         (candidate :warm-ssd :ssd 192
                    {:candidate/queue-ms 0
                     :candidate/prefix-load-ms 4
                     :candidate/gpu-restore-ms 3
                     :candidate/prefill-ms-per-token 0.2
                     :candidate/first-token-ms 2})]
        result (-> (sim/make-sim workers)
                   (sim/submit request candidates)
                   (sim/run-until-response :demo/request 1000))]
    {:response (sim/response result :demo/request)
     :selected-worker
     (get-in result [:sim/router :router/requests :demo/request
                     :assignment/candidate :candidate/worker-id])
     :logical-time (:sim/time result)
     :trace (:sim/trace result)}))

(defn- observation
  [worker-id node model overrides]
  (merge
   {:worker/id worker-id
    :worker/node node
    :worker/epoch 0
    :worker/models #{model}
    :worker/queue-ms 0
    :worker/page-size 16
    :worker/free-pages 128
    :worker/evictable-pages 0
    :worker/max-context 8192
    :worker/prefill-ms-per-token 0.2
    :worker/first-token-ms 2
    :worker/gpu-restore-bytes-per-ms 1048576
    :worker/tier-throughput-bytes-per-ms {:ssd 1048576}
    :worker/tier-fixed-ms {:ssd 0}}
   overrides))

(defn run-database-simulation
  "Derive two-worker candidates from Datahike placement and worker observations.

  The in-memory catalog contains four durable exact-prefix chunks. Three have a
  ready SSD replica on `warm-ssd`; `busy-gpu` reports the full prefix resident
  but a longer queue. No tensor payload is needed because this demonstrates the
  decision-grade database/control path, then executes it under logical time."
  []
  (let [model "gemma-fixture-v1"
        request {:request/id :demo/database-request
                 :request/model-fingerprint model
                 :request/tokens (vec (range 257))
                 :request/max-new-tokens 8
                 :request/simulated-output [42 43 44 45 46 47 48 49]}
        config {:store {:backend :memory :id (random-uuid)}
                :schema-flexibility :write
                :keep-history? false
                :value-caps :default}
        connection (catalog/ensure-database! config)
        descriptors (chunk/plan (:request/tokens request) 256 64)
        chunk-bytes 1048576]
    (try
      (catalog/put-chunks!
       connection model
       (mapv #(assoc % :store-key (random-uuid) :bytes chunk-bytes)
             descriptors))
      (doseq [descriptor (take 3 descriptors)]
        (placement/announce-replica!
         connection
         {:model-fingerprint model
          :prefix-hash (:chunk/prefix-hash descriptor)
          :node "warm-ssd" :tier :ssd :state :kv.replica/ready
          :store-key (random-uuid) :bytes chunk-bytes}))
      (let [tail (peek descriptors)
            observations
            [(observation
              :busy-gpu "busy-gpu" model
              {:worker/queue-ms 45
               :worker/gpu-prefixes
               {[model (:chunk/prefix-hash tail)]
                {:continuation-id :demo/resident-prefix
                 :token-count 256
                 :bytes (* 4 chunk-bytes)}}})
             (observation :warm-ssd "warm-ssd" model {})]
            derived (candidates/candidates
                     @connection request observations {:chunk-size 64})
            workers
            {:busy-gpu {:worker/epoch 0 :worker/models #{model}
                        :worker/free-pages 128 :worker/evictable-pages 0}
             :warm-ssd {:worker/epoch 0 :worker/models #{model}
                        :worker/free-pages 128 :worker/evictable-pages 0}}
            result (-> (sim/make-sim workers)
                       (sim/submit request derived)
                       (sim/run-until-response :demo/database-request 1000))]
        {:response (sim/response result :demo/database-request)
         :candidates
         (mapv #(select-keys % [:candidate/worker-id :candidate/cache-tier
                                :candidate/cached-token-count
                                :estimate/ttft-ms])
               derived)
         :selected-worker
         (get-in result [:sim/router :router/requests :demo/database-request
                         :assignment/candidate :candidate/worker-id])
         :logical-time (:sim/time result)
         :trace (:sim/trace result)})
      (finally
        (d/release connection)
        (d/delete-database config)))))
