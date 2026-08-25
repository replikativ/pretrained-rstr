(ns pretrained.continuation-page-pool-test
  (:require [clojure.test :refer [deftest is testing]]
            [pretrained.attention-state :as attention-state]
            [pretrained.continuation :as continuation]
            [pretrained.continuation.page-pool :as page-pool]
            [raster.gpu.core :as gpu]))

(def ^:private model
  {:n-layers 2 :n-kv 1 :head-dim 2})

(def ^:private layout
  (attention-state/layout model))

(defn- fixture-pool
  [state]
  (page-pool/->DevicePagePool
   ::session layout 4 8 :half
   {[:key 0] :pool-k0
    [:key 1] :pool-k1
    [:value 0] :pool-v0
    [:value 1] :pool-v1}
   state))

(deftest routes-share-full-pages-and-copy-on-write-the-tail
  (let [copies (atom [])
        views (atom [])
        pool (fixture-pool
              (atom {:free (apply sorted-set (range 8))
                     :refcounts {}
                     :routes {}}))]
    (with-redefs [gpu/buffer-view
                  (fn [_ key opts]
                    (let [view {:key key :opts opts}]
                      (swap! views conj view)
                      view))
                  gpu/copy-range!
                  (fn [_ source destination opts]
                    (swap! copies conj [source destination opts])
                    destination)]
      (let [root (page-pool/allocate-route! pool :root 6)
            fork (page-pool/fork-route! pool :root :fork)]
        (is (= [0 1] (:pages root) (:pages fork)))
        (is (= 6 (page-pool/free-page-count pool)))
        (let [reservation (page-pool/reserve-append! pool :fork)]
          (testing "the shared partial tail is copied as one coherent model page"
            (is (= 1 (:replaced-page reservation)))
            (is (= 2 (:physical-page reservation)))
            (is (= 2 (:page-offset reservation)))
            (is (= 4 (count @copies)))
            (is (every? #(= {:elements 8} (nth % 2)) @copies)))
          (let [committed (page-pool/commit-append! pool :fork reservation)]
            (is (= [0 2] (:pages committed)))
            (is (= 7 (:token-count committed)))
            (is (= [0 1] (:pages (page-pool/route pool :root))))))
        (is (page-pool/release-route! pool :fork))
        (is (= 6 (page-pool/free-page-count pool)))
        (is (page-pool/release-route! pool :root))
        (is (= 8 (page-pool/free-page-count pool)))))))

(deftest aborted-append-restores-the-shared-route
  (let [pool (fixture-pool
              (atom {:free (apply sorted-set (range 8))
                     :refcounts {}
                     :routes {}}))]
    (with-redefs [gpu/buffer-view (fn [_ key opts] {:key key :opts opts})
                  gpu/copy-range! (fn [_ _ destination _] destination)]
      (page-pool/allocate-route! pool :root 3)
      (page-pool/fork-route! pool :root :fork)
      (let [reservation (page-pool/reserve-append! pool :fork)
            restored (page-pool/abort-append! pool :fork reservation)]
        (is (= [0] (:pages restored)))
        (is (= 3 (:token-count restored)))
        (is (= 2 (get-in @(:state pool) [:refcounts 0])))
        (is (= 7 (page-pool/free-page-count pool)))))))

(deftest durable-chunk-scatters-across-noncontiguous-physical-pages
  (let [uploads (atom nil)
        pool (fixture-pool
              (atom {:free (sorted-set 0 2 4 6)
                     :refcounts {1 1, 3 1, 5 1, 7 1}
                     :routes {:continuation
                              {:continuation-id :continuation
                               :pages [5 1]
                               :token-count 7
                               :start-position 0}}}))
        descriptor {:chunk/start 2
                    :chunk/token-count 4
                    :chunk/layout (continuation/model-layout model)}
        ;; 2 slabs × 2 layers × 4 tokens × 2 elements/token
        payload (float-array (map float (range 32)))]
    (with-redefs [gpu/buffer-view (fn [_ key opts] {:key key :opts opts})
                  gpu/upload-ranges!
                  (fn [_ entries] (reset! uploads entries) (mapv second entries))]
      (page-pool/restore-chunk! pool :continuation descriptor payload)
      (testing "each slab/layer splits at the logical page boundary"
        (is (= 8 (count @uploads)))
        (is (= [:pool-k0 :pool-k0 :pool-k1 :pool-k1
                :pool-v0 :pool-v0 :pool-v1 :pool-v1]
               (mapv #(get-in % [0 :key]) @uploads))))
      (testing "logical tokens 2-3 target physical page 5; 4-5 target page 1"
        (is (= [40 8 40 8 40 8 40 8]
               (mapv #(quot (get-in % [0 :opts :byte-offset]) 2) @uploads)))
        (is (= [4 0 4 0 4 0 4 0]
               (mapv #(get-in % [2 :dst-element]) @uploads))))
      (testing "source slices follow slab/layer payload order"
        (is (= [0 4 8 12 16 20 24 28]
               (mapv #(get-in % [2 :src-element]) @uploads)))
        (is (every? #(= 4 (get-in % [2 :elements])) @uploads))))))

(deftest dense-routes-compose-unrelated-continuations-into-a-batch
  (let [pool (fixture-pool
              (atom {:free (sorted-set 6 7)
                     :refcounts {0 1, 1 1, 2 1, 3 1, 4 1, 5 1}
                     :routes {:a {:continuation-id :a :pages [4 2]
                                  :token-count 6 :start-position 0}
                              :b {:continuation-id :b :pages [5 0 3 1]
                                  :token-count 13 :start-position 8}}}))
        values (page-pool/dense-route-values pool [:a :b])]
    (is (= [4 2 -1 -1, 5 0 3 1] (vec (:page-table values))))
    (is (= [6 13] (vec (:lengths values))))
    (is (= [0 8] (vec (:start-positions values))))
    (is (= 4 (:pages-per-sequence values)))))

(deftest append-commits-only-after-the-complete-transfer
  (let [pool (fixture-pool
              (atom {:free (apply sorted-set (range 8))
                     :refcounts {}
                     :routes {}}))
        uploads (atom nil)
        rows {[:key 0] (float-array [1 2])
              [:key 1] (float-array [3 4])
              [:value 0] (float-array [5 6])
              [:value 1] (float-array [7 8])}]
    (page-pool/allocate-route! pool :continuation 0)
    (with-redefs [gpu/upload-ranges!
                  (fn [_ entries] (reset! uploads entries) (mapv second entries))]
      (let [resident-route (page-pool/append-token! pool :continuation rows)]
        (is (= 1 (:token-count resident-route)))
        (is (= [0] (:pages resident-route)))
        (is (= [:pool-k0 :pool-k1 :pool-v0 :pool-v1]
               (mapv first @uploads)))
        (is (every? #(= 2 (get-in % [2 :elements])) @uploads))
        (is (every? #(instance? (Class/forName "[S") (second %)) @uploads))))
    (with-redefs [gpu/upload-ranges! (fn [& _] (throw (ex-info "device fault" {})))]
      (is (thrown-with-msg? clojure.lang.ExceptionInfo #"device fault"
                            (page-pool/append-token! pool :continuation rows)))
      (is (= 1 (:token-count (page-pool/route pool :continuation)))))))

(deftest leases-pin-an-immutable-route-snapshot-through-copy-on-write
  (let [pool (fixture-pool
              (atom {:free (apply sorted-set (range 8))
                     :refcounts {}
                     :leases {}
                     :routes {}}))]
    (with-redefs [gpu/buffer-view (fn [_ key opts] {:key key :opts opts})
                  gpu/copy-range! (fn [_ _ destination _] destination)]
      (page-pool/allocate-route! pool :request 3)
      (let [lease (page-pool/acquire-lease! pool [:request])
            reservation (page-pool/reserve-append! pool :request)]
        (is (= 0 (:replaced-page reservation))
            "the lease makes the live partial tail copy-on-write")
        (page-pool/commit-append! pool :request reservation)
        (is (= [1] (:pages (page-pool/route pool :request))))
        (is (= [0] (-> lease :routes first :pages))
            "the scheduled page-table snapshot remains immutable")
        (is (page-pool/release-route! pool :request))
        (is (= 7 (page-pool/free-page-count pool))
            "the old page remains pinned after logical eviction")
        (is (= [0] (vec (:page-table
                         (page-pool/leased-dense-route-values pool lease)))))
        (is (page-pool/release-lease! pool lease))
        (is (= 8 (page-pool/free-page-count pool)))))))

(deftest prospective-leases-see-a-whole-batch-only-after-exact-reservation
  (let [pool (fixture-pool
              (atom {:free (apply sorted-set (range 8))
                     :refcounts {}
                     :leases {}
                     :routes {}}))]
    (page-pool/allocate-route! pool :a 3)
    (page-pool/allocate-route! pool :b 4)
    (let [entries (mapv (fn [continuation-id]
                          {:continuation-id continuation-id
                           :reservation (page-pool/reserve-append!
                                         pool continuation-id)})
                        [:a :b])
          lease (page-pool/acquire-prospective-lease! pool entries)
          values (page-pool/leased-dense-route-values
                  pool lease {:pages-per-sequence 2})]
      (is (= [4 5] (vec (:lengths values))))
      (is (= [0 -1, 1 2] (vec (:page-table values))))
      (is (= [3 4] (mapv :token-count [(page-pool/route pool :a)
                                       (page-pool/route pool :b)]))
          "the live routes stay uncommitted")
      (is (thrown-with-msg? clojure.lang.ExceptionInfo #"stale"
                            (page-pool/acquire-prospective-lease!
                             pool [(assoc (first entries) :reservation {})])))
      (is (= [4 5]
             (mapv :token-count (page-pool/commit-appends! pool entries))))
      (is (page-pool/release-lease! pool lease)))))

(deftest batched-abort-validates-every-reservation-before-changing-routes
  (let [pool (fixture-pool
              (atom {:free (apply sorted-set (range 8))
                     :refcounts {}
                     :leases {}
                     :routes {}}))]
    (page-pool/allocate-route! pool :a 0)
    (page-pool/allocate-route! pool :b 0)
    (let [entries (mapv (fn [continuation-id]
                          {:continuation-id continuation-id
                           :reservation (page-pool/reserve-append!
                                         pool continuation-id)})
                        [:a :b])]
      (is (thrown-with-msg? clojure.lang.ExceptionInfo #"stale"
                            (page-pool/abort-appends!
                             pool (assoc-in entries [1 :reservation] {}))))
      (is (every? :pending [(page-pool/route pool :a)
                            (page-pool/route pool :b)]))
      (is (= [0 0]
             (mapv :token-count (page-pool/abort-appends! pool entries))))
      (is (= 8 (page-pool/free-page-count pool))))))
