(ns pretrained.continuation-page-pool-test
  (:require [clojure.test :refer [deftest is testing]]
            [pretrained.attention-state :as attention-state]
            [pretrained.continuation :as continuation]
            [pretrained.continuation.block-transfer :as block-transfer]
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
                    :chunk/layout
                    (assoc-in (continuation/model-layout model)
                              [:attention-state :dtype] :float16)}
        ;; 2 slabs × 2 layers × 4 tokens × 2 elements/token
        payload (short-array (map #(Float/floatToFloat16 (float %)) (range 32)))]
    (with-redefs [gpu/buffer-view (fn [_ key opts] {:key key :opts opts})
                  gpu/submit-upload-ranges!
                  (fn [_ entries] (reset! uploads entries) entries)
                  gpu/await-event! (fn [_ entries] (mapv second entries))
                  gpu/event-measurement
                  (fn [& _] {:direction :upload :timing-source :device-event
                             :asynchronous? true :bytes 64 :commands 8
                             :elapsed-ns 40 :submit-host-ns 4 :host-wall-ns 50})
                  gpu/release-event! (fn [& _] nil)]
      (page-pool/restore-chunk! pool :continuation descriptor payload)
      (testing "each slab/layer splits at the logical page boundary"
        (is (= 8 (count @uploads)))
        (is (= [:pool-k0 :pool-k0 :pool-k1 :pool-k1
                :pool-v0 :pool-v0 :pool-v1 :pool-v1]
               (mapv #(get-in % [0 :key]) @uploads))))
      (testing "logical tokens 2-3 target physical page 5; 4-5 target page 1"
        (is (= [44 8 44 8 44 8 44 8]
               (mapv #(quot (get-in % [0 :opts :byte-offset]) 2) @uploads)))
        (is (= [0 0 0 0 0 0 0 0]
               (mapv #(get-in % [2 :dst-element]) @uploads))))
      (testing "source slices follow slab/layer payload order"
        (is (= [0 4 8 12 16 20 24 28]
               (mapv #(get-in % [2 :src-element]) @uploads)))
        (is (every? #(= 4 (get-in % [2 :elements])) @uploads)))
      (is (= 64 (get-in (page-pool/transfer-stats pool)
                        [:counters [:upload :device-event true] :bytes]))))))

(deftest durable-chunk-coalesces-contiguous-physical-pages
  (let [uploads (atom nil)
        pool (fixture-pool
              (atom {:free (sorted-set 0 3 4 5 6 7)
                     :refcounts {1 1, 2 1}
                     :routes {:continuation
                              {:continuation-id :continuation
                               :pages [1 2]
                               :token-count 8
                               :start-position 0}}}))
        descriptor {:chunk/start 2
                    :chunk/token-count 4
                    :chunk/layout
                    (assoc-in (continuation/model-layout model)
                              [:attention-state :dtype] :float16)}
        payload (short-array 32)]
    (with-redefs [gpu/buffer-view (fn [_ key opts] {:key key :opts opts})
                  gpu/submit-upload-ranges!
                  (fn [_ entries] (reset! uploads entries) entries)
                  gpu/await-event! (fn [_ entries] (mapv second entries))
                  gpu/event-measurement
                  (fn [& _] {:direction :upload :timing-source :device-event
                             :asynchronous? true :bytes 64 :commands 4
                             :elapsed-ns 30 :submit-host-ns 3 :host-wall-ns 40})
                  gpu/release-event! (fn [& _] nil)]
      (page-pool/restore-chunk! pool :continuation descriptor payload)
      (is (= 4 (count @uploads))
          "one contiguous run is submitted for each slab/layer")
      (is (= [12 12 12 12]
             (mapv #(quot (get-in % [0 :opts :byte-offset]) 2) @uploads)))
      (is (= [0 8 16 24] (mapv #(get-in % [2 :src-element]) @uploads)))
      (is (every? #(= 8 (get-in % [2 :elements])) @uploads)))))

(deftest fragmented-durable-chunk-uses-dense-resident-block-staging
  (let [opened (atom [])
        runs (atom [])
        submissions (atom [])
        pool (fixture-pool
              (atom {:free (sorted-set 3 5 7)
                     :refcounts {0 1, 1 1, 2 1, 4 1, 6 1}
                     :routes {:continuation
                              {:continuation-id :continuation
                               :pages [0 2 4 6 1]
                               :token-count 20
                               :start-position 0}}}))
        descriptor {:chunk/start 0
                    :chunk/token-count 20
                    :chunk/layout
                    (assoc-in (continuation/model-layout model)
                              [:attention-state :dtype] :float16)}
        payload (short-array (* 2 2 20 2))]
    (with-redefs [block-transfer/open!
                  (fn [_ _ _ _ _ nblocks]
                    (let [engine {:nblocks nblocks}]
                      (swap! opened conj engine)
                      engine))
                  block-transfer/index-buffer-key
                  (fn [engine] [:indices (:nblocks engine)])
                  block-transfer/staging-buffer-key
                  (fn [engine slab layer]
                    [:staging (:nblocks engine) slab layer])
                  block-transfer/run!
                  (fn [engine direction]
                    (swap! runs conj [(:nblocks engine) direction]))
                  gpu/submit-upload-ranges!
                  (fn [_ entries]
                    (let [event {:direction :upload :entries (vec entries)}]
                      (swap! submissions conj event)
                      event))
                  gpu/submit-download-ranges!
                  (fn [_ entries]
                    (let [event {:direction :download :entries (vec entries)}]
                      (swap! submissions conj event)
                      event))
                  gpu/await-event!
                  (fn [_ event] (mapv second (:entries event)))
                  gpu/event-measurement
                  (fn [_ event]
                    {:direction (:direction event)
                     :timing-source :host-monotonic
                     :asynchronous? false
                     :bytes 160
                     :commands (count (:entries event))
                     :elapsed-ns 40 :submit-host-ns 40 :host-wall-ns 40})
                  gpu/release-event! (fn [& _] nil)]
      (is (= {:page-blocks 5 :token-capacity 20
              :staging-bytes 320 :index-bytes 20 :workspace-bytes 340}
             (page-pool/prepare-block-transfer! pool 17)))
      (page-pool/restore-chunk! pool :continuation descriptor payload)
      (let [chunk (page-pool/export-chunk
                   pool :continuation "fragmented-fixture" descriptor)]
        (is (= (alength payload) (alength ^shorts (:chunk/payload chunk)))))
      (is (= [{:nblocks 5}] @opened))
      (is (= [[5 :scatter] [5 :gather]] @runs))
      (is (= [[:upload 5] [:upload 1] [:download 4]]
             (mapv (juxt :direction (comp count :entries)) @submissions)))
      (is (= #{5}
             (set (keys (:block-transfer-engines @(:state pool)))))))))

(deftest resident-pages-gather-into-the-portable-durable-chunk-layout
  (let [downloads (atom nil)
        pool (fixture-pool
              (atom {:free (sorted-set 0 2 4 6)
                     :refcounts {1 1, 3 1, 5 1, 7 1}
                     :leases {}
                     :routes {:continuation
                              {:continuation-id :continuation
                               :pages [5 1]
                               :token-count 7
                               :start-position 0}}}))
        descriptor {:chunk/start 2 :chunk/token-count 4}]
    (with-redefs [gpu/buffer-view (fn [_ key opts] {:key key :opts opts})
                  gpu/submit-download-ranges!
                  (fn [_ entries] (reset! downloads entries) entries)
                  gpu/await-event!
                  (fn [_ entries]
                    (reset! downloads entries)
                    (doseq [[_ ^shorts destination {:keys [dst-element elements]}] entries
                            index (range elements)]
                      (let [destination-index (+ dst-element index)]
                        (aset destination destination-index
                              (Float/floatToFloat16
                               (float (inc destination-index))))))
                    (mapv second entries))
                  gpu/event-measurement
                  (fn [& _] {:direction :download :timing-source :host-monotonic
                             :asynchronous? false :bytes 64 :commands 8
                             :elapsed-ns 60 :submit-host-ns 60 :host-wall-ns 60})
                  gpu/release-event! (fn [& _] nil)]
      (let [chunk (page-pool/export-chunk
                   pool :continuation "fixture-paged-v1" descriptor)]
        (is (= 3 (:chunk/version chunk)))
        (is (= "fixture-paged-v1" (:chunk/model-fingerprint chunk)))
        (is (= :float16 (get-in chunk [:chunk/layout :dtype])))
        (is (= (mapv #(Float/floatToFloat16 (float %)) (range 1 33))
               (vec (:chunk/payload chunk))))
        (is (= 8 (count @downloads))
            "every slab/layer splits at the physical page boundary")
        (is (zero? (:active-leases (page-pool/stats pool))))
        (is (= :host-monotonic
               (get-in (page-pool/transfer-stats pool) [:last :timing-source])))))))

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
