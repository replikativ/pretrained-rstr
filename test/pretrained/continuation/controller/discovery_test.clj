(ns pretrained.continuation.controller.discovery-test
  (:require [clojure.test :refer [deftest is]]
            [pretrained.continuation.controller.discovery :as discovery]))

(defn- observation
  [worker-id epoch sequence queue-ms]
  {:worker/id worker-id
   :worker/node (name worker-id)
   :worker/epoch epoch
   :worker/sequence sequence
   :worker/models #{"fixture-v1"}
   :worker/queue-ms queue-ms
   :worker/page-size 16
   :worker/free-pages 8
   :worker/evictable-pages 0
   :worker/max-context 1024
   :worker/prefill-ms-per-token 1
   :worker/first-token-ms 1
   :worker/gpu-restore-bytes-per-ms 1000
   :worker/tier-throughput-bytes-per-ms {}})

(deftest latest-epoch-and-sequence-win-deterministically
  (let [initial (-> (discovery/initial-state)
                    (discovery/observe (observation :worker-b 0 1 10))
                    (discovery/observe (observation :worker-a 0 2 20)))
        stale (discovery/observe initial (observation :worker-a 0 1 99))
        duplicate-version
        (discovery/observe stale (observation :worker-a 0 2 98))
        restarted
        (discovery/observe duplicate-version (observation :worker-a 1 0 3))]
    (is (= initial stale))
    (is (= stale duplicate-version))
    (is (= [:worker-a :worker-b]
           (mapv :worker/id (discovery/observations restarted))))
    (is (= 3 (:worker/queue-ms
              (first (discovery/observations restarted)))))
    (is (= [:worker-a]
           (mapv :worker/id
                 (discovery/observations
                  (discovery/remove-worker restarted :worker-b)))))))
