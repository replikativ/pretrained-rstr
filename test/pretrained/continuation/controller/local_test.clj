(ns pretrained.continuation.controller.local-test
  (:require [clojure.test :refer [deftest is]]
            [pretrained.attention-state :as attention-state]
            [pretrained.continuation.controller.local :as local]
            [pretrained.continuation.page-pool :as page-pool]))

(def ^:private model-fingerprint "fixture-model-v1")

(defn- fixture-pool
  [physical-pages]
  (page-pool/->DevicePagePool
   ::session (attention-state/layout {:n-layers 1 :n-kv 1 :head-dim 2})
   2 physical-pages :half
   {[:key 0] :pool-k0, [:value 0] :pool-v0}
   (atom {:free (apply sorted-set (range physical-pages))
          :refcounts {} :routes {}})))

(defn- request
  []
  {:request/id :request-a
   :request/model-fingerprint model-fingerprint
   :request/tokens [1 2 3 4]
   :request/max-new-tokens 2})

(defn- candidate
  [free-pages]
  {:candidate/worker-id :worker-a
   :candidate/worker-epoch 0
   :candidate/cache-tier :ssd
   :candidate/cached-token-count 2
   :candidate/cached-bytes 32
   :candidate/queue-ms 0
   :candidate/prefix-load-ms 1
   :candidate/gpu-restore-ms 1
   :candidate/prefill-ms-per-token 1
   :candidate/first-token-ms 1
   :candidate/page-size 2
   :candidate/free-pages free-pages
   :candidate/evictable-pages 0
   :candidate/max-context 32})

(defn- offer
  [free-pages]
  {:event/type :assignment/offered
   :assignment/id [:request-a 1]
   :assignment/worker-epoch 0
   :assignment/request (request)
   :assignment/candidate (candidate free-pages)})

(defn- test-controller
  [pool pending sent handlers]
  (local/open-controller
   pool
   {:worker/id :worker-a :worker/epoch 0
    :worker/models #{model-fingerprint}
    :worker/free-pages 0 :worker/evictable-pages 0}
   {:handlers (merge
               {:worker/restore-prefix (fn [_] nil)
                :worker/prefill-suffix (fn [_] nil)
                :worker/decode (fn [_] {:tokens [9 10]})}
               handlers)
    :send! #(swap! sent conj %)
    :submit! (fn [task]
               (swap! pending conj task)
               task)
    :cancel! (constantly nil)
    :close-submit! (constantly nil)}))

(defn- run-next!
  [pending]
  (let [task (first @pending)]
    (swap! pending subvec 1)
    (task)))

(deftest accepted-work-holds-capacity-through-the-operation-chain
  (let [pool (fixture-pool 4)
        pending (atom [])
        sent (atom [])
        operations (atom [])
        observe (fn [result]
                  (fn [effect]
                    (swap! operations conj effect)
                    result))
        controller
        (test-controller
         pool pending sent
         {:worker/restore-prefix (observe {:cached-token-count 1})
          :worker/prefill-suffix (observe nil)
          :worker/decode (observe {:tokens [9 10]})})]
    (try
      (local/handle-event! controller (offer 4))
      (is (= [true] (mapv :event/accepted? @sent)))
      (is (= 3 (:reserved-pages (page-pool/stats pool))))
      (is (= 1 (count @pending)))
      (run-next! pending)
      (run-next! pending)
      (run-next! pending)
      (is (= [:worker/restore-prefix :worker/prefill-suffix :worker/decode]
             (mapv :effect/op @operations)))
      (is (every? page-pool/capacity-reservation?
                  (map :worker/capacity-reservation @operations)))
      (is (= :completed (get-in (local/state controller)
                                [:worker/assignments [:request-a 1]
                                 :assignment/phase])))
      (is (zero? (:reserved-pages (page-pool/stats pool))))
      (is (= {:status :completed :tokens [9 10] :cached-token-count 1}
             (:event/result (last @sent))))
      (finally
        (.close controller)))))

(deftest actual-page-pressure-declines-a-stale-offer-before-io
  (let [pool (fixture-pool 2)
        pending (atom [])
        sent (atom [])
        controller (test-controller pool pending sent {})]
    (try
      (local/handle-event! controller (offer 4))
      (is (= false (:event/accepted? (first @sent))))
      (is (= :insufficient-pages (:event/reason (first @sent))))
      (is (empty? @pending))
      (is (= 2 (page-pool/free-page-count pool)))
      (finally
        (.close controller)))))

(deftest gpu-prefix-is-forked-before-incremental-capacity-is-reserved
  (let [pool (fixture-pool 4)
        _ (page-pool/allocate-route! pool :shared-prefix 2)
        pending (atom [])
        sent (atom [])
        controller (test-controller pool pending sent {})
        event (-> (offer 3)
                  (assoc-in [:assignment/request :request/continuation-id]
                            :request-route)
                  (assoc-in [:assignment/candidate :candidate/cache-tier] :gpu)
                  (assoc-in [:assignment/candidate
                             :candidate/source-continuation-id]
                            :shared-prefix))]
    (try
      (local/handle-event! controller event)
      (is (true? (:event/accepted? (first @sent))))
      (is (= (:pages (page-pool/route pool :shared-prefix))
             (:pages (page-pool/route pool :request-route))))
      (is (= 2 (:reserved-pages (page-pool/stats pool)))
          "only growth beyond the shared resident prefix is reserved")
      (is (= :prefilling
             (get-in (local/state controller)
                     [:worker/assignments [:request-a 1] :assignment/phase])))
      (local/handle-event!
       controller
       {:event/type :assignment/cancelled
        :assignment/id [:request-a 1]
        :request/id :request-a})
      (run-next! pending)
      (is (zero? (:reserved-pages (page-pool/stats pool))))
      (finally
        (.close controller)))))

(deftest observation-combines-device-capacity-and-exact-prefix-identity
  (let [pool (fixture-pool 4)
        prefix (random-uuid)
        _ (page-pool/allocate-route!
           pool :resident 2
           {:policy {:durable? true
                     :model-fingerprint model-fingerprint
                     :prefix-hash prefix}})
        controller (test-controller pool (atom []) (atom []) {})]
    (try
      (let [result
            (local/observation
             controller
             {:worker/node "worker-a"
              :worker/queue-ms 3
              :worker/transfer-capabilities
              {:backend :simulated
               :live-overlap-eligible? true}})]
        (is (= :worker-a (:worker/id result)))
        (is (= #{model-fingerprint} (:worker/models result)))
        (is (= 3 (:worker/free-pages result)))
        (is (= 1 (:worker/evictable-pages result)))
        (is (true? (get-in result
                           [:worker/transfer-capabilities
                            :live-overlap-eligible?])))
        (is (= {:continuation-id :resident
                :token-count 2
                :bytes 16}
               (get-in result [:worker/gpu-prefixes
                               [model-fingerprint prefix]]))))
      (finally
        (.close controller)))))

(deftest cancellation-releases-unclaimed-projected-capacity
  (let [pool (fixture-pool 4)
        pending (atom [])
        sent (atom [])
        cancelled (atom [])
        cancelled-operations (atom [])
        controller (assoc (test-controller pool pending sent {})
                          :cancel! #(swap! cancelled conj %)
                          :cancel-operation!
                          #(swap! cancelled-operations conj %))]
    (try
      (local/handle-event! controller (offer 4))
      (is (= 1 (count @pending)))
      (local/handle-event!
       controller
       {:event/type :assignment/cancelled
        :assignment/id [:request-a 1]
        :request/id :request-a})
      (is (= 1 (count @cancelled))
          "cancellation reaches the active handler task")
      (is (= [[:request-a 1]]
             (mapv :assignment/id @cancelled-operations)))
      (is (= 3 (:reserved-pages (page-pool/stats pool)))
          "capacity remains fenced while accepted I/O may still be running")
      (run-next! pending)
      (is (zero? (:reserved-pages (page-pool/stats pool))))
      (is (= 4 (page-pool/free-page-count pool)))
      (finally
        (.close controller)))))
