(ns pretrained.continuation-placement-test
  (:require [clojure.test :refer [deftest is testing]]
            [datahike.api :as d]
            [pretrained.continuation.catalog :as catalog]
            [pretrained.continuation.chunk :as chunk]
            [pretrained.continuation.placement :as placement])
  (:import [java.util Date]))

(defn- memory-config
  []
  {:store {:backend :memory :id (random-uuid)}
   :schema-flexibility :write
   :keep-history? false
   :value-caps :default})

(deftest placement-facts-produce-reconciliation-actions
  (let [config (memory-config)
        connection (catalog/ensure-database! config)
        descriptor (first (chunk/plan [1 2] 2 2))
        prefix (:chunk/prefix-hash descriptor)
        model "fixture-v1"
        events (atom [])
        listener (placement/listen! connection "worker-b" #(swap! events conj %))]
    (try
      (catalog/put-chunks! connection model
                           [(assoc descriptor :store-key (random-uuid) :bytes 128)])
      (placement/request! connection
                          {:model-fingerprint model :prefix-hash prefix
                           :node "worker-b" :tier :ssd :priority 10
                           :owner "inference-7"})
      (testing "an unplaced demand resolves its immutable catalog chunk"
        (let [plan (placement/reconciliation-plan @connection "worker-b")
              action (first (:actions plan))]
          (is (empty? (:satisfied plan)))
          (is (= :ensure-local (:action action)))
          (is (= 128 (get-in action [:chunk :kv/bytes])))
          (is (empty? (:sources action)))))
      (placement/announce-replica!
       connection {:model-fingerprint model :prefix-hash prefix
                   :node "worker-a" :tier :ssd :state :kv.replica/ready
                   :store-key (random-uuid) :path "/worker-a/chunk" :bytes 128})
      (testing "a ready remote replica becomes a source candidate"
        (let [action (first (:actions
                            (placement/reconciliation-plan @connection "worker-b")))]
          (is (= ["worker-a"] (mapv :kv/replica-node (:sources action))))))
      (placement/announce-replica!
       connection {:model-fingerprint model :prefix-hash prefix
                   :node "worker-b" :tier :ssd :state :kv.replica/copying})
      (testing "a local copy in progress suppresses duplicate actions"
        (let [plan (placement/reconciliation-plan @connection "worker-b")]
          (is (empty? (:actions plan)))
          (is (= 1 (count (:in-progress plan))))))
      (placement/announce-replica!
       connection {:model-fingerprint model :prefix-hash prefix
                   :node "worker-b" :tier :ssd :state :kv.replica/ready
                   :store-key (random-uuid) :path "/worker-b/chunk" :bytes 128})
      (testing "the desired local tier satisfies the demand"
        (let [plan (placement/reconciliation-plan @connection "worker-b")]
          (is (empty? (:actions plan)))
          (is (= 1 (count (:satisfied plan))))))
      (testing "placement writes produce lightweight tx notifications"
        (is (= 4 (count @events)))
        (is (every? #(= "worker-b" (:node %)) @events)))
      (finally
        (placement/unlisten! connection listener)
        (d/release connection)
        (d/delete-database config)))))

(deftest demands-are-idempotent-prioritized-and-expirable
  (let [config (memory-config)
        connection (catalog/ensure-database! config)
        model "fixture-v1"
        low-prefix (random-uuid)
        high-prefix (random-uuid)]
    (try
      (placement/request! connection
                          {:model-fingerprint model :prefix-hash low-prefix
                           :node "worker-a" :tier :ssd :priority 1})
      (placement/request! connection
                          {:model-fingerprint model :prefix-hash high-prefix
                           :node "worker-a" :tier :gpu :priority 20})
      (placement/request! connection
                          {:model-fingerprint model :prefix-hash low-prefix
                           :node "worker-a" :tier :ssd :priority 5})
      (is (= [20 5]
             (mapv :kv/demand-priority
                   (placement/demands @connection "worker-a"))))
      (placement/request! connection
                          {:model-fingerprint model :prefix-hash (random-uuid)
                           :node "worker-a" :tier :ssd :priority 100
                           :expires-at (Date. 0)})
      (is (= [20 5]
             (mapv :kv/demand-priority
                   (placement/demands @connection "worker-a"))))
      (is (= (placement/demand-id model low-prefix "worker-a" :ssd)
             (placement/demand-id model low-prefix "worker-a" :ssd)))
      (finally
        (d/release connection)
        (d/delete-database config)))))
