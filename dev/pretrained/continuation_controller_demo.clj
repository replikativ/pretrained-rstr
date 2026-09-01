(ns pretrained.continuation-controller-demo
  "Model-free deterministic cluster-continuation demonstration."
  (:require [pretrained.continuation.controller.sim :as sim]))

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
