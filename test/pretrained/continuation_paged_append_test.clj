(ns pretrained.continuation-paged-append-test
  (:require [clojure.test :refer [deftest is testing]]
            [pretrained.attention-state :as attention-state]
            [pretrained.continuation.page-pool :as page-pool]
            [pretrained.continuation.paged-append :as paged-append]
            [raster.compiler.ir.kernel-graph :as kernel-graph]
            [raster.gpu.core :as gpu]))

(def ^:private layout
  (attention-state/layout {:n-layers 2 :n-kv 1 :head-dim 4}))

(defn- pool
  []
  (page-pool/->DevicePagePool
   ::session layout 4 8 :half
   {[:key 0] :k0, [:key 1] :k1, [:value 0] :v0, [:value 1] :v1}
   (atom {:free (apply sorted-set (range 8))
          :refcounts {}
          :leases {}
          :routes {}})))

(deftest reservation-batches-have-stable-physical-slots-and-one-publication-point
  (let [page-pool (pool)]
    (page-pool/allocate-route! page-pool :a 3)
    (page-pool/allocate-route! page-pool :b 4)
    (let [batch (paged-append/reserve-batch! page-pool [:a :b])]
      (is (paged-append/append-batch? batch))
      (is (= [3 8] (vec (paged-append/slot-values batch))))
      (is (= [:a :b] (mapv :continuation-id
                           (paged-append/reservation-entries batch))))
      (is (= [3 4] (mapv :token-count [(page-pool/route page-pool :a)
                                       (page-pool/route page-pool :b)])))
      (is (= [4 5] (mapv :token-count (paged-append/commit-batch! batch))))
      (is (thrown-with-msg? clojure.lang.ExceptionInfo #"already been completed"
                            (paged-append/abort-batch! batch))))))

(deftest reservation-batches-reject-duplicate-lanes-and-can-abort
  (let [page-pool (pool)]
    (page-pool/allocate-route! page-pool :a 0)
    (is (thrown-with-msg? clojure.lang.ExceptionInfo #"must be unique"
                          (paged-append/reserve-batch! page-pool [:a :a])))
    (let [batch (paged-append/reserve-batch! page-pool [:a])]
      (is (= [0] (mapv :token-count (paged-append/abort-batch! batch))))
      (is (= 8 (page-pool/free-page-count page-pool))))))

(deftest partial-batch-reservation-failure-rolls-back-earlier-lanes
  (let [page-pool (page-pool/->DevicePagePool
                   ::session layout 4 1 :half
                   {[:key 0] :k0, [:key 1] :k1
                    [:value 0] :v0, [:value 1] :v1}
                   (atom {:free (sorted-set 0)
                          :refcounts {} :leases {} :routes {}}))]
    (page-pool/allocate-route! page-pool :a 0)
    (page-pool/allocate-route! page-pool :b 0)
    (is (thrown-with-msg? clojure.lang.ExceptionInfo #"insufficient free capacity"
                          (paged-append/reserve-batch! page-pool [:a :b])))
    (is (= 1 (page-pool/free-page-count page-pool)))
    (is (every? #(nil? (:pending (page-pool/route page-pool %))) [:a :b]))))

(deftest reference-plan-binds-resident-rows-directly-to-raster-assignment
  (let [page-pool (pool)
        key-view (Object.)
        value-view (Object.)]
    (with-redefs [gpu/resident-buffer-view? (constantly true)]
      (let [plan (paged-append/reference-plan
                  page-pool
                  {:id :append-fixture
                   :key-prefix "append-fixture"
                   :layer 1
                   :batch-size 2
                   :key-view key-view
                   :value-view value-view})
            problem (:problem plan)]
        (is (= :fp32-to-fp16-reference (:strategy plan)))
        (is (= 2 (:batch-size problem)))
        (is (= 4 (:key-elements-per-token problem)
               (:value-elements-per-token problem)))
        (is (identical? key-view
                        (get (:bindings plan) (:key-rows problem))))
        (is (identical? value-view
                        (get (:bindings plan) (:value-rows problem))))
        (is (= :k1 (get (:bindings plan) (:key-pages problem))))
        (is (= :v1 (get (:bindings plan) (:value-pages problem))))
        (is (kernel-graph/kernel-graph? (:graph plan)))))))

(deftest runner-completes-device-writes-without-implicitly-committing-routes
  (let [page-pool (pool)
        batch (do
                (page-pool/allocate-route! page-pool :a 0)
                (page-pool/allocate-route! page-pool :b 0)
                (paged-append/reserve-batch! page-pool [:a :b]))
        uploaded (atom nil)
        released (atom [])
        event (Object.)
        problem
        (with-redefs [gpu/resident-buffer-view? (constantly true)]
          (:problem
           (paged-append/reference-plan
            page-pool
            {:layer 0 :batch-size 2
             :key-view (Object.) :value-view (Object.)})))
        runner (paged-append/->PagedAppendRunner
                ::session page-pool 0 problem ::handle :slots ::graph
                (atom {:closed? false :pending nil :batch nil}))]
    (with-redefs [gpu/upload! (fn [_ key values]
                                (reset! uploaded [key (vec values)]))
                  gpu/submit-kernel-graph! (fn [_ _] event)
                  gpu/await-event! (fn [_ actual] (is (identical? event actual)))
                  gpu/release-event! (fn [_ actual] (swap! released conj actual))]
      (paged-append/load-batch! runner batch)
      (is (= [:slots [0 4]] @uploaded))
      (is (identical? event (paged-append/submit! runner)))
      (is (identical? batch (paged-append/await! runner event)))
      (testing "GPU completion and logical publication are separate"
        (is (= [0 0] (mapv :token-count [(page-pool/route page-pool :a)
                                         (page-pool/route page-pool :b)])))
        (is (= [1 1] (mapv :token-count
                           (paged-append/commit-batch! batch)))))
      (is (= [event] @released)))))
