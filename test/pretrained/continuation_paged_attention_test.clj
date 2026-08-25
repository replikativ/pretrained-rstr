(ns pretrained.continuation-paged-attention-test
  (:require [clojure.test :refer [deftest is testing]]
            [pretrained.attention-state :as attention-state]
            [pretrained.continuation.page-pool :as page-pool]
            [pretrained.continuation.paged-attention :as paged-attention]
            [raster.compiler.ir.attention :as attention]
            [raster.compiler.ir.kernel-graph :as kernel-graph]
            [raster.gpu.core :as gpu]))

(def ^:private model
  {:n-layers 2 :n-kv 2 :head-dim 8})

(defn- pool
  []
  (page-pool/->DevicePagePool
   ::session (attention-state/layout model) 4 16 :half
   {[:key 0] :k0, [:key 1] :k1, [:value 0] :v0, [:value 1] :v1}
   (atom {:free (apply sorted-set (range 16))
          :refcounts {}
          :routes {}})))

(deftest reference-plan-binds-page-pools-without-inventing-another-kernel-abi
  (let [plan (paged-attention/reference-plan
              (pool)
              {:id :fixture
               :key-prefix "fixture"
               :layer 1
               :batch-size 2
               :total-query-tokens 2
               :q-heads 4
               :kv-heads 2
               :qk-head-dim 8
               :value-head-dim 8
               :pages-per-sequence 4})
        problem (:problem plan)]
    (is (= :fp16-reference (:strategy plan)))
    (is (attention/attention-problem? problem))
    (is (= :dense-paged (attention/route-kind (:route problem))))
    (is (= :page-major (:k-layout problem) (:v-layout problem)))
    (is (= :k1 (get (:bindings plan) (get-in plan [:ids :key-pages]))))
    (is (= :v1 (get (:bindings plan) (get-in plan [:ids :value-pages]))))
    (is (kernel-graph/kernel-graph? (:graph plan)))
    (is (= #{:half :int}
           (set (map first (vals (:allocations plan))))))))

(deftest resident-views-compose-attention-with-adjacent-device-graphs
  (let [page-pool (pool)
        query-view (Object.)
        output-view (Object.)]
    (with-redefs [gpu/resident-buffer-view? (constantly true)]
      (let [plan (paged-attention/reference-plan
                  page-pool
                  {:id :resident-fixture
                   :key-prefix "resident-fixture"
                   :layer 0
                   :batch-size 2
                   :total-query-tokens 2
                   :q-heads 4
                   :kv-heads 2
                   :qk-head-dim 8
                   :value-head-dim 8
                   :pages-per-sequence 2
                   :query-dtype :float
                   :output-dtype :float
                   :query-view query-view
                   :output-view output-view})]
        (testing "caller-owned tensor views replace only private tensor allocations"
          (is (identical? query-view
                          (get (:bindings plan) (get-in plan [:ids :query]))))
          (is (identical? output-view
                          (get (:bindings plan) (get-in plan [:ids :output]))))
          (is (= :float (get-in plan [:problem :q-dtype])))
          (is (= :float (get-in plan [:problem :output-dtype])))
          (is (= #{:int} (set (map first (vals (:allocations plan))))))
          (is (= (set (keys (:allocations plan))) (:owned-buffer-keys plan))))))))

(deftest resident-query-and-output-avoid-host-tensor-transfers
  (let [page-pool (pool)
        output-view (Object.)
        uploads (atom nil)
        submitted (Object.)
        plan (paged-attention/reference-plan
              page-pool
              {:id :resident-fixture
               :key-prefix "resident-fixture"
               :layer 0
               :batch-size 2
               :total-query-tokens 2
               :q-heads 4
               :kv-heads 2
               :qk-head-dim 8
               :value-head-dim 8
               :pages-per-sequence 2})
        plan (assoc-in plan [:options :query-view] (Object.))
        plan (assoc-in plan [:options :output-view] output-view)
        runner (paged-attention/->PagedAttentionRunner
                ::session page-pool (:problem plan) ::handle
                {:query :q
                 :query-row-offsets :q-offsets
                 :query-positions :q-positions
                 :page-table :page-table
                 :lengths :lengths
                 :start-positions :starts
                 :output :out}
                ::graph
                (atom {:closed? false :pending nil :lease nil :plan plan}))]
    (page-pool/allocate-route! page-pool :a 5)
    (page-pool/allocate-route! page-pool :b 3)
    (with-redefs [gpu/upload-ranges!
                  (fn [_ entries] (reset! uploads entries) (mapv second entries))
                  gpu/submit-kernel-graph! (fn [_ _] submitted)
                  gpu/await-event! (fn [_ event] (is (identical? submitted event)))
                  gpu/release-event! (fn [_ _] nil)
                  gpu/download (fn [& _] (throw (ex-info "unexpected tensor download" {})))]
      (paged-attention/load-batch!
       runner
       {:continuation-ids [:a :b]
        :row-offsets [0 1 2]
        :positions [4 2]})
      (is (= [:q-offsets :q-positions :page-table :lengths :starts]
             (mapv first @uploads)))
      (is (identical? output-view
                      (paged-attention/await! runner
                                              (paged-attention/submit! runner))))
      (is (zero? (:active-leases (page-pool/stats page-pool)))))))

(deftest runner-composes-page-routes-and-uses-raster-events
  (let [page-pool (pool)
        uploads (atom nil)
        submitted (Object.)
        released (atom [])
        output (short-array [1 2 3 4])
        runner (paged-attention/->PagedAttentionRunner
                ::session page-pool
                (:problem
                 (paged-attention/reference-plan
                  page-pool
                  {:id :fixture
                   :key-prefix "fixture"
                   :layer 0
                   :batch-size 2
                   :total-query-tokens 2
                   :q-heads 4
                   :kv-heads 2
                   :qk-head-dim 8
                   :value-head-dim 8
                   :pages-per-sequence 2}))
                ::handle
                {:query :q
                 :query-row-offsets :q-offsets
                 :query-positions :q-positions
                 :page-table :page-table
                 :lengths :lengths
                 :start-positions :starts
                 :output :out}
                ::graph
                (atom {:closed? false
                       :pending nil
                       :plan {:options {:batch-size 2
                                        :total-query-tokens 2
                                        :pages-per-sequence 2}}}))]
    (page-pool/allocate-route! page-pool :a 5)
    (page-pool/allocate-route! page-pool :b 3 {:start-position 9})
    (with-redefs [gpu/upload-ranges!
                  (fn [_ entries] (reset! uploads entries) (mapv second entries))
                  gpu/submit-kernel-graph! (fn [_ _] submitted)
                  gpu/await-event! (fn [_ event] (is (identical? submitted event)))
                  gpu/release-event! (fn [_ event] (swap! released conj event))
                  gpu/download (fn [_ key] (is (= :out key)) output)]
      (paged-attention/load-batch!
       runner
       {:continuation-ids [:a :b]
        :query-values (repeat (* 2 4 8) 0.25)
        :row-offsets [0 1 2]
        :positions [4 11]})
      (testing "one validated transfer batch updates query and route descriptors"
        (is (= [:q :q-offsets :q-positions :page-table :lengths :starts]
               (mapv first @uploads)))
        (is (= [0 1 2 -1] (vec (second (nth @uploads 3)))))
        (is (= [5 3] (vec (second (nth @uploads 4)))))
        (is (= [0 9] (vec (second (nth @uploads 5))))))
      (let [event (paged-attention/submit! runner)]
        (is (identical? submitted event))
        (is (identical? output (paged-attention/await! runner event)))
        (is (= [submitted] @released))
        (is (nil? (:pending @(:state runner))))))))

(deftest runner-can-load-prospective-post-append-routes
  (let [page-pool (pool)
        uploads (atom nil)
        runner (paged-attention/->PagedAttentionRunner
                ::session page-pool
                (:problem
                 (paged-attention/reference-plan
                  page-pool
                  {:id :prospective
                   :key-prefix "prospective"
                   :layer 0
                   :batch-size 2
                   :total-query-tokens 2
                   :q-heads 4
                   :kv-heads 2
                   :qk-head-dim 8
                   :value-head-dim 8
                   :pages-per-sequence 2}))
                ::handle
                {:query :q
                 :query-row-offsets :q-offsets
                 :query-positions :q-positions
                 :page-table :page-table
                 :lengths :lengths
                 :start-positions :starts
                 :output :out}
                ::graph
                (atom {:closed? false :pending nil :lease nil
                       :plan {:options {:batch-size 2
                                        :total-query-tokens 2
                                        :pages-per-sequence 2}}}))]
    (page-pool/allocate-route! page-pool :a 5)
    (page-pool/allocate-route! page-pool :b 3)
    (let [entries (mapv (fn [continuation-id]
                          {:continuation-id continuation-id
                           :reservation (page-pool/reserve-append!
                                         page-pool continuation-id)})
                        [:a :b])]
      (with-redefs [gpu/upload-ranges!
                    (fn [_ values] (reset! uploads values) (mapv second values))]
        (paged-attention/load-batch!
         runner
         {:continuation-ids [:a :b]
          :append-reservations entries
          :query-values (repeat (* 2 4 8) 0.25)
          :row-offsets [0 1 2]
          :positions [5 3]})
        (is (= [6 4] (vec (second (nth @uploads 4))))
            "attention sees each successfully reserved token")
        (is (= [5 3] (mapv :token-count
                           [(page-pool/route page-pool :a)
                            (page-pool/route page-pool :b)]))
            "reservation loading does not publish the append")
        (is (thrown-with-msg? clojure.lang.ExceptionInfo #"do not align"
                              (paged-attention/load-batch!
                               runner
                               {:continuation-ids [:a :b]
                                :append-reservations (vec (reverse entries))
                                :query-values (repeat (* 2 4 8) 0.25)
                                :row-offsets [0 1 2]
                                :positions [5 3]}))))
      (page-pool/release-lease! page-pool (:lease @(:state runner)))
      (page-pool/abort-appends! page-pool entries))))
