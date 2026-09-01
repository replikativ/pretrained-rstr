(ns pretrained.continuation.controller.kabel-test
  (:require [clojure.core.async :as async]
            [clojure.test :refer [deftest is]]
            [datahike.api :as d]
            [pretrained.attention-state :as attention-state]
            [pretrained.continuation.catalog :as catalog]
            [pretrained.continuation.controller.kabel :as controller-kabel]
            [pretrained.continuation.page-pool :as page-pool]
            [pretrained.continuation.paged-runtime :as paged-runtime]))

(def ^:private model-fingerprint "fixture-model-v1")

(defn- memory-config
  []
  {:store {:backend :memory :id (random-uuid)}
   :schema-flexibility :write
   :keep-history? false
   :value-caps :default})

(defn- fixture-pool
  ([] (fixture-pool 8))
  ([physical-pages]
   (page-pool/->DevicePagePool
    ::session (attention-state/layout {:n-layers 1 :n-kv 1 :head-dim 2})
    2 physical-pages :half
    {[:key 0] :pool-k0, [:value 0] :pool-v0}
    (atom {:free (apply sorted-set (range physical-pages))
           :refcounts {} :routes {}}))))

(defn- measurements
  [node queue-ms prefill-ms]
  {:worker/node node
   :worker/queue-ms queue-ms
   :worker/max-context 64
   :worker/prefill-ms-per-token prefill-ms
   :worker/first-token-ms 1
   :worker/gpu-restore-bytes-per-ms 1000
   :worker/transfer-capabilities {}
   :worker/tier-throughput-bytes-per-ms {}
   :worker/object-store? false})

(defn- handlers
  [tokens]
  {:worker/restore-prefix (fn [_] nil)
   :worker/prefill-suffix (fn [_] nil)
   :worker/decode (fn [_] {:tokens tokens})})

(defn- worker-endpoint
  [worker-id node queue-ms prefill-ms tokens heartbeat-ms]
  (controller-kabel/open-worker-endpoint
   (fixture-pool)
   {:worker/id worker-id
    :worker/epoch 0
    :worker/models #{model-fingerprint}
    :worker/free-pages 0
    :worker/evictable-pages 0}
   {:handlers (handlers tokens)
    :measurements (measurements node queue-ms prefill-ms)
    :heartbeat-ms heartbeat-ms
    :on-error! #(throw %)}))

(defn- attach!
  [router worker]
  (let [to-router (async/chan 64)
        to-worker (async/chan 64)]
    ((controller-kabel/router-middleware router)
     [nil nil [to-router to-worker]])
    ((controller-kabel/worker-middleware worker)
     [nil nil [to-worker to-router]])
    [to-router to-worker]))

(defn- await-value
  [timeout-ms operation]
  (let [deadline (+ (System/nanoTime) (* 1000000 timeout-ms))]
    (loop []
      (if-let [value (operation)]
        value
        (if (< (System/nanoTime) deadline)
          (do (Thread/sleep 5) (recur))
          (throw (ex-info "Timed out waiting for controller state"
                          {:timeout-ms timeout-ms})))))))

(deftest live-observations-drive-offer-acknowledgement-and-result
  (let [config (memory-config)
        connection (catalog/ensure-database! config)
        delivered (atom [])
        errors (atom [])
        router (controller-kabel/open-router-endpoint
                connection
                {:deliver! #(swap! delivered conj %)
                 :heartbeat-timeout-ms 1000
                 :offer-timeout-ms 200
                 :chunk-size 2
                 :on-error! #(swap! errors conj %)})
        slow (worker-endpoint :slow "slow" 20 10 [40] 100)
        fast (worker-endpoint :fast "fast" 0 1 [41 42] 100)
        slow-channels (attach! router slow)
        fast-channels (attach! router fast)
        request {:request/id :live-request
                 :request/model-fingerprint model-fingerprint
                 :request/tokens [1 2 3 4]
                 :request/max-new-tokens 2}]
    (try
      (await-value 1000
                   #(when (= 2 (count (controller-kabel/observations router)))
                      true))
      (controller-kabel/submit! router request)
      (let [response (await-value 1000 #(first @delivered))
            assignment (get-in (controller-kabel/router-state router)
                               [:router/requests :live-request])]
        (is (= :completed (:response/type response)))
        (is (= [41 42] (get-in response [:response/value :tokens])))
        (is (= :fast
               (get-in assignment
                       [:assignment/candidate :candidate/worker-id])))
        (is (= :completed (:assignment/phase assignment)))
        (is (empty? @errors)))
      (finally
        (doseq [channel (concat slow-channels fast-channels)]
          (async/close! channel))
        (.close slow)
        (.close fast)
        (.close router)
        (d/release connection)
        (d/delete-database config)))))

(deftest heartbeat-expiry-removes-a-connected-but-silent-worker
  (let [config (memory-config)
        connection (catalog/ensure-database! config)
        router (controller-kabel/open-router-endpoint
                connection
                {:deliver! (constantly nil)
                 :heartbeat-timeout-ms 40
                 :on-error! #(throw %)})
        worker (worker-endpoint :worker-a "worker-a" 0 1 [9] 10000)
        channels (attach! router worker)]
    (try
      (await-value 500 #(first (controller-kabel/observations router)))
      (is (empty? (await-value
                   500
                   #(when (empty? (controller-kabel/observations router)) []))))
      (finally
        (doseq [channel channels]
          (async/close! channel))
        (.close worker)
        (.close router)
        (d/release connection)
        (d/delete-database config)))))

(deftest concurrent-controller-assignments-share-paged-runtime-lanes
  (let [config (memory-config)
        connection (catalog/ensure-database! config)
        delivered (atom [])
        errors (atom [])
        route-counts (atom {})
        submissions (atom [])
        fake-decoder {:decode-state {:batch-size 2 :maxpos 64}}
        runtime
        (paged-runtime/open-runtime
         fake-decoder
         {:prime-lanes! (fn [& _] nil)
          :step-lanes!
          (fn [_ work]
            (Thread/sleep 2)
            (swap! submissions conj work)
            (mapv (fn [{:keys [continuation-id position] :as item}]
                    (swap! route-counts update continuation-id (fnil inc 0))
                    (assoc item :token (+ 100 position)))
                  work))
          :route-token-count (fn [_ id] (get @route-counts id 0)
                               )})
        submission (paged-runtime/controller-submission runtime 8)
        worker
        (controller-kabel/open-worker-endpoint
         (fixture-pool 16)
         {:worker/id :batched-worker
          :worker/epoch 0
          :worker/models #{model-fingerprint}
          :worker/free-pages 0
          :worker/evictable-pages 0}
         (merge
          submission
          {:handlers
           {:worker/restore-prefix
            (fn [value]
              (paged-runtime/run-operation!
               runtime (:assignment/id value)
               #(do (swap! route-counts assoc
                            (get-in value
                                    [:assignment/request
                                     :request/continuation-id]) 0)
                    {:ok? true})))
            :worker/prefill-suffix #(paged-runtime/prefill! runtime %)
            :worker/decode #(paged-runtime/decode! runtime %)}
           :measurements (measurements "batched-worker" 0 1)
           :heartbeat-ms 100
           :on-error! #(swap! errors conj %)}))
        router
        (controller-kabel/open-router-endpoint
         connection
         {:deliver! #(swap! delivered conj %)
          :heartbeat-timeout-ms 1000
          :offer-timeout-ms 500
          :chunk-size 2
          :on-error! #(swap! errors conj %)})
        channels (attach! router worker)
        request (fn [id]
                  {:request/id id
                   :request/continuation-id id
                   :request/model-fingerprint model-fingerprint
                   :request/tokens (vec (range 10))
                   :request/max-new-tokens 2})]
    (try
      (await-value 1000 #(first (controller-kabel/observations router)))
      (controller-kabel/submit! router (request :batch-a))
      (controller-kabel/submit! router (request :batch-b))
      (await-value 3000 #(when (= 2 (count @delivered)) true))
      (is (= #{:batch-a :batch-b}
             (set (map :request/id @delivered))))
      (is (every? #(= :completed (:response/type %)) @delivered))
      (is (some #(= #{:batch-a :batch-b}
                    (set (map :continuation-id %)))
                @submissions)
          "accepted assignments execute in one sparse fixed batch")
      (is (empty? @errors))
      (finally
        (doseq [channel channels]
          (async/close! channel))
        (.close worker)
        (.close runtime)
        (.close router)
        (d/release connection)
        (d/delete-database config)))))
