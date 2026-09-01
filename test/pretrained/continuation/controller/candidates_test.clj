(ns pretrained.continuation.controller.candidates-test
  (:require [clojure.test :refer [deftest is testing]]
            [datahike.api :as d]
            [pretrained.continuation.catalog :as catalog]
            [pretrained.continuation.chunk :as chunk]
            [pretrained.continuation.controller.candidates :as candidates]
            [pretrained.continuation.placement :as placement]))

(def ^:private model "fixture-model-v1")

(def ^:private request
  {:request/id :request-a
   :request/model-fingerprint model
   :request/tokens [1 2 3 4 5 6 7]
   :request/max-new-tokens 2})

(defn- memory-config
  []
  {:store {:backend :memory :id (random-uuid)}
   :schema-flexibility :write
   :keep-history? false
   :value-caps :default})

(defn- observation
  [overrides]
  (merge
   {:worker/id :worker-a
    :worker/node "worker-a"
    :worker/epoch 0
    :worker/models #{model}
    :worker/queue-ms 0
    :worker/page-size 2
    :worker/free-pages 16
    :worker/evictable-pages 0
    :worker/max-context 32
    :worker/prefill-ms-per-token 10
    :worker/first-token-ms 1
    :worker/gpu-restore-bytes-per-ms 1000
    :worker/tier-throughput-bytes-per-ms {:ssd 1000 :object 1}
    :worker/tier-fixed-ms {}}
   overrides))

(defn- with-catalog
  [operation]
  (let [config (memory-config)
        connection (catalog/ensure-database! config)
        descriptors (chunk/plan (:request/tokens request) 6 2)]
    (try
      (catalog/put-chunks!
       connection model
       (mapv #(assoc % :store-key (random-uuid) :bytes 200) descriptors))
      (operation connection descriptors)
      (finally
        (d/release connection)
        (d/delete-database config)))))

(deftest local-ready-chain-becomes-an-ssd-candidate
  (with-catalog
    (fn [connection descriptors]
      (doseq [descriptor (take 2 descriptors)]
        (placement/announce-replica!
         connection
         {:model-fingerprint model
          :prefix-hash (:chunk/prefix-hash descriptor)
          :node "worker-a" :tier :ssd :state :kv.replica/ready
          :store-key (random-uuid) :bytes 200}))
      (let [result (first (candidates/candidates
                           @connection request
                           [(observation
                             {:worker/transfer-capabilities
                              {:live-overlap-eligible? true}})]
                           {:chunk-size 2}))]
        (is (= :ssd (:candidate/cache-tier result)))
        (is (= 4 (:candidate/cached-token-count result)))
        (is (= 400 (:candidate/cached-bytes result)))
        (is (true? (get-in result
                           [:candidate/transfer-capabilities
                            :live-overlap-eligible?])))))))

(deftest shorter-local-prefix-can-beat-a-long-object-prefix
  (with-catalog
    (fn [connection descriptors]
      (placement/announce-replica!
       connection
       {:model-fingerprint model
        :prefix-hash (:chunk/prefix-hash (first descriptors))
        :node "worker-a" :tier :ssd :state :kv.replica/ready
        :store-key (random-uuid) :bytes 200})
      (let [result (first (candidates/candidates
                           @connection request
                           [(observation {:worker/object-store? true})]
                           {:chunk-size 2}))]
        (testing "routing minimizes predicted TTFT instead of forcing longest reuse"
          (is (= :ssd (:candidate/cache-tier result)))
          (is (= 2 (:candidate/cached-token-count result))))))))

(deftest gpu-observation-names-the-route-that-the-worker-will-fork
  (with-catalog
    (fn [connection descriptors]
      (let [descriptor (second descriptors)
            prefix (:chunk/prefix-hash descriptor)
            result
            (first
             (candidates/candidates
              @connection request
              [(observation
                {:worker/gpu-prefixes
                 {[model prefix]
                  {:continuation-id :resident-prefix
                   :token-count 4 :bytes 400}}})]
              {:chunk-size 2}))]
        (is (= :gpu (:candidate/cache-tier result)))
        (is (= 4 (:candidate/cached-token-count result)))
        (is (= :resident-prefix
               (:candidate/source-continuation-id result)))))))

(deftest stale-gpu-placement-retains-a-durable-worker-fallback
  (with-catalog
    (fn [connection descriptors]
      (doseq [descriptor (take 2 descriptors)]
        (placement/announce-replica!
         connection
         {:model-fingerprint model
          :prefix-hash (:chunk/prefix-hash descriptor)
          :node "worker-a" :tier :ssd :state :kv.replica/ready
          :store-key (random-uuid) :bytes 200}))
      (let [prefix (:chunk/prefix-hash (second descriptors))
            result
            (candidates/candidates
             @connection request
             [(observation
               {:worker/gpu-prefixes
                {[model prefix]
                 {:continuation-id :resident-prefix
                  :token-count 4 :bytes 400}}})]
             {:chunk-size 2})]
        (is (= [:gpu :ssd :ssd :none]
               (mapv :candidate/cache-tier result)))
        (is (= [4 4 2 0]
               (mapv :candidate/cached-token-count result)))))))

(deftest infeasible-workers-are-omitted
  (with-catalog
    (fn [connection _]
      (is (empty?
           (candidates/candidates
            @connection request
            [(observation {:worker/online? false})]
            {:chunk-size 2})))
      (is (empty?
           (candidates/candidates
            @connection request
            [(observation {:worker/models #{}})]
            {:chunk-size 2}))))))
