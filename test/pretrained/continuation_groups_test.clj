(ns pretrained.continuation-groups-test
  "Retention groups: layout derivation, per-group storage, part selection,
  window floors, and restores that load only what a boundary needs."
  (:require [clojure.test :refer [deftest is testing]]
            [datahike.api :as d]
            [pretrained.attention-state :as attention-state]
            [pretrained.continuation :as continuation]
            [pretrained.continuation.block-transfer :as block-transfer]
            [pretrained.continuation.catalog :as catalog]
            [pretrained.continuation.chunk :as chunk]
            [pretrained.continuation.chunk-store :as chunk-store]
            [pretrained.continuation.manager :as manager]
            [pretrained.continuation.manifest :as manifest]
            [pretrained.continuation.page-pool :as page-pool]
            [pretrained.continuation.parts :as parts]
            [raster.compiler.ir.numerical-state :as numerical-state])
  (:import [java.nio.file Files]
           [java.nio.file.attribute FileAttribute]))

;; Two layers: layer 1 is global, layer 0 slides over 8 tokens.
(def ^:private model
  {:n-layers 2 :n-kv 1 :head-dim 2
   :desc {:flags {:global-layer-pattern 2 :sliding-window {:size 8}}}})

(def ^:private fingerprint "groups-test/2-layer-window-8")

(deftest windowed-models-derive-groups
  (let [layout (attention-state/layout model)]
    (is (= [{:id :global :extent {:kind :token} :members [[:key 1] [:value 1]]}
            {:id :window-8 :extent {:kind :window :size 8}
             :members [[:key 0] [:value 0]]}]
           (:groups layout)))
    (is (= [{:slab :key :layer 0 :element-offset 0 :elements 6}
            {:slab :value :layer 0 :element-offset 6 :elements 6}]
           (attention-state/group-payload-plan layout (second (:groups layout)) 3)))
    (testing "slabs not indexed by model layer cannot be grouped"
      (is (thrown-with-msg?
           clojure.lang.ExceptionInfo #"indexed by model layer"
           (attention-state/layout
            (assoc-in model [:desc :attention-state]
                      {:slabs [{:name :latent :tensor-key :continuation/latent
                                :buffer-prefix "lc" :count 1
                                :elements-per-token 2}]})))))))

(defn- values [n offset]
  (let [a (float-array n)]
    (dotimes [i n] (aset a i (float (+ offset (/ i 100.0)))))
    a))

(defn- fixture [processed]
  (let [elements (* processed 2)]
    {:continuation/backend :cpu
     :continuation/model-fingerprint fingerprint
     :continuation/layout (continuation/model-layout model)
     :continuation/processed-count processed
     :continuation/pending-token processed
     :continuation/tokens (mapv long (range (inc processed)))
     :continuation/keys [(values elements 0.0) (values elements 10.0)]
     :continuation/values [(values elements 20.0) (values elements 30.0)]}))

(deftest cpu-chunks-split-rows-by-group
  (let [state (fixture 12)
        descriptor (second (:chunks (chunk/continuation-plan state 4)))
        [global window] (chunk/cpu-tensor-chunks state descriptor)]
    (is (= [:global :window-8] [(:chunk/group global) (:chunk/group window)]))
    (is (= [[:key 1] [:value 1]] (mapv (juxt :slab :layer) (:chunk/slabs global))))
    (is (= (vec (take 8 (drop 8 (seq (values 24 10.0)))))
           (vec (take 8 (seq ^floats (:chunk/payload global)))))
        "the global part starts with layer 1's key rows 4..7")
    (is (= (vec (take 8 (drop 8 (seq (values 24 0.0)))))
           (vec (take 8 (seq ^floats (:chunk/payload window)))))
        "the window part starts with layer 0's key rows 4..7")
    (is (thrown-with-msg? clojure.lang.ExceptionInfo #"stored per group"
                          (chunk/cpu-tensor-chunk state descriptor)))))

(defn- with-manager [f]
  (let [config {:store {:backend :memory :id (random-uuid)}
                :schema-flexibility :write :keep-history? false :value-caps :default}
        directory (Files/createTempDirectory "groups-test-" (make-array FileAttribute 0))
        cache (manager/open-manager config directory {:chunk-size 4})]
    (try
      (f cache)
      (finally
        (.close cache)
        (d/delete-database config)))))

(deftest nodes-publish-every-part-together
  (with-manager
    (fn [cache]
      (let [stored (manager/checkpoint-cpu-chunks! cache (fixture 20))
            database @(:connection cache)
            entry (catalog/lookup-chunk database fingerprint
                                        (:chunk/prefix-hash (first stored)))]
        (is (= 5 (count stored)))
        (is (nil? (:kv/store-key entry))
            "a multi-group node has no node-level blob to misload")
        (is (= [:global :window-8] (mapv :group (parts/entry-parts entry))))
        (is (every? :store-key (parts/entry-parts entry)))
        (testing "retracting a node retracts its parts"
          (let [part-ids (set (map :kv.part/id (:kv/parts entry)))
                part-present? (fn [db]
                                (set (d/q '[:find [?id ...]
                                            :in $ [?id ...]
                                            :where [_ :kv.part/id ?id]]
                                          db part-ids)))]
            (is (= 2 (count part-ids)))
            (is (= part-ids (part-present? @(:connection cache))))
            (catalog/retract! (:connection cache) (:kv/id entry))
            (is (empty? (part-present? @(:connection cache))))))))))

(deftest boundary-selection-loads-only-the-attended-window
  (with-manager
    (fn [cache]
      (let [stored (manager/checkpoint-cpu-chunks! cache (fixture 20))
            database @(:connection cache)
            entries (manifest/chain database fingerprint
                                    (:chunk/prefix-hash (peek stored)))
            selection (parts/boundary-selection
                       (attention-state/layout model) entries)]
        ;; The token at 20 attends to rows 13..19 of the window layer.
        (is (= 20 (:boundary selection)))
        (is (= {:window-8 12} (:window-floors selection)))
        (is (= [[:global] [:global] [:global] [:global :window-8] [:global :window-8]]
               (mapv #(mapv :group (:parts %)) (:chunks selection))))
        (let [certified (manifest/continuation-manifest
                         database fingerprint (:chunk/prefix-hash (peek stored)))
              [global window] (get-in certified [:manifest :fields])
              plan (manifest/load-plan certified)]
          (is (numerical-state/certified-state? (numerical-state/verify! certified)))
          (is (= [2 20 2] (get-in global [:value :shape])))
          (is (= [2 8 2] (get-in window [:value :shape])))
          (is (= {:token-origin 12 :members [[:key 0] [:value 0]]}
                 (:coordinate-space window)))
          (is (= {:window-8 12} (:window-floors plan)))
          (is (= 7 (count (:parts plan))))
          (is (= (parts/selected-bytes selection)
                 (reduce + 0 (map :bytes (:parts plan))))
              "the manifest and the selection agree on what a boundary loads"))))))

(defn- synthetic-chain
  "Root-first catalog entries of `n` chunks of `size` tokens, both groups."
  [n size]
  (vec (for [i (range n)]
         {:kv/prefix-hash (keyword (str "h" i))
          :kv/start-token (* i size)
          :kv/token-count size
          :kv/parts [{:kv.part/group :global :kv.part/store-key (random-uuid) :kv.part/bytes 1}
                     {:kv.part/group :window-8 :kv.part/store-key (random-uuid) :kv.part/bytes 1}]})))

(deftest window-selection-at-chunk-edges
  ;; With w = 8 the token at b attends rows [b - 7, b). A 2-token tail chunk
  ;; puts boundaries next to the floor's chunk edge.
  (let [layout (attention-state/layout model)
        chain (fn [boundary]
                (let [full (quot boundary 4)
                      tail (rem boundary 4)]
                  (cond-> (synthetic-chain full 4)
                    (pos? tail)
                    (conj {:kv/prefix-hash :tail :kv/start-token (* full 4) :kv/token-count tail
                           :kv/parts [{:kv.part/group :global :kv.part/store-key (random-uuid)
                                       :kv.part/bytes 1}
                                      {:kv.part/group :window-8 :kv.part/store-key (random-uuid)
                                       :kv.part/bytes 1}]}))))
        window-starts (fn [selection]
                        (vec (for [{:keys [start parts]} (:chunks selection)
                                   :when (some #(= :window-8 (:group %)) parts)]
                               start)))]
    (testing "b = 18: the first attended row 11 lies in chunk [8, 12)"
      (let [selection (parts/boundary-selection layout (chain 18))]
        (is (= {:window-8 8} (:window-floors selection)))
        (is (= [8 12 16] (window-starts selection)))))
    (testing "b = 19: the first attended row 12 starts chunk [12, 16)"
      (let [selection (parts/boundary-selection layout (chain 19))]
        (is (= {:window-8 12} (:window-floors selection)))
        (is (= [12 16] (window-starts selection)))))
    (testing "b = 20: rows 13..19"
      (is (= {:window-8 12}
             (:window-floors (parts/boundary-selection layout (chain 20))))))
    (testing "a boundary inside the window keeps every row"
      (let [selection (parts/boundary-selection layout (chain 8))]
        (is (= {:window-8 0} (:window-floors selection)))
        (is (= [0 4] (window-starts selection)))))
    (testing "the global group is always complete"
      (is (every? (fn [{:keys [parts]}] (some #(= :global (:group %)) parts))
                  (:chunks (parts/boundary-selection layout (chain 19))))))))

(defn- fixture-pool [physical-pages]
  (page-pool/->DevicePagePool
   ::session (attention-state/layout model) 4 physical-pages :half
   {[:key 0] :k0 [:key 1] :k1 [:value 0] :v0 [:value 1] :v1}
   (atom {:free (apply sorted-set (range physical-pages))
          :refcounts {} :routes {}})))

(deftest window-floors-guard-forks-and-captures
  (let [pool (fixture-pool 16)]
    (page-pool/allocate-route! pool :restored 20)
    (page-pool/set-window-floors! pool :restored {:window-8 12})
    (testing "forks at boundaries whose next token attends above the floor"
      (is (page-pool/fork-allowed? pool :restored 20))
      (is (page-pool/fork-allowed? pool :restored 19))
      (is (not (page-pool/fork-allowed? pool :restored 18))
          "the token at 18 attends row 11, which was never restored")
      (is (thrown-with-msg? clojure.lang.ExceptionInfo #"below a window floor"
                            (page-pool/fork-route! pool :restored :early 16)))
      (let [fork (page-pool/fork-route! pool :restored :late 19)]
        (is (= {:window-8 12} (:window-floors fork))
            "a fork inherits its source's floors")))
    (testing "a route without floors forks anywhere"
      (page-pool/allocate-route! pool :fresh 12)
      (is (page-pool/fork-allowed? pool :fresh 4)))
    (testing "capture refuses window rows below the floor"
      (let [lease (#'page-pool/acquire-lease! pool [:restored])]
        (try
          (is (thrown-with-msg?
               clojure.lang.ExceptionInfo #"never restored"
               (#'page-pool/export-chunk-plan
                pool lease :restored
                {:chunk/start 8 :chunk/token-count 4 :chunk/group :window-8})))
          (is (map? (#'page-pool/export-chunk-plan
                     pool lease :restored
                     {:chunk/start 8 :chunk/token-count 4 :chunk/group :global}))
              "the global group holds every row")
          (finally
            (page-pool/release-lease! pool lease)))))))

(deftest block-engines-are-per-group
  (let [pool (fixture-pool 16)
        opened (atom [])]
    (with-redefs [block-transfer/open!
                  (fn [_ _ _ _ _ nblocks members]
                    (swap! opened conj [nblocks members])
                    {:nblocks nblocks})]
      (page-pool/prepare-block-transfer! pool 8))
    (is (= [[2 [[:key 1] [:value 1]]] [2 [[:key 0] [:value 0]]]] @opened)
        "a scatter compiled for one group writes only that group's rows")
    (is (= #{[:global 2] [:window-8 2]}
           (set (keys (:block-transfer-engines @(:state pool))))))))

(deftest paged-restore-loads-the-manifest-plan-and-sets-floors
  (with-manager
    (fn [cache]
      (let [pool (fixture-pool 16)
            durable (page-pool/durable-layout pool)
            tokens (mapv long (range 21))
            plan (chunk/plan tokens 20 4)
            nodes
            (mapv (fn [descriptor]
                    (mapv (fn [group]
                            (let [rows (reduce + 0 (map :elements
                                                        (attention-state/group-payload-plan
                                                         (:layout pool) group
                                                         (:chunk/token-count descriptor))))]
                              (assoc descriptor
                                     :chunk/version 3
                                     :chunk/model-fingerprint fingerprint
                                     :chunk/layout durable
                                     :chunk/group (:id group)
                                     :chunk/slabs []
                                     :chunk/payload (short-array rows))))
                          (:groups (:layout pool))))
                  plan)
            _ (#'manager/publish-chunks!
               cache fingerprint (mapv #(#'manager/persist-node! cache %) nodes))
            restored (atom [])]
        (with-redefs [page-pool/restore-chunk!
                      (fn [_ id descriptor _]
                        (swap! restored conj
                               [id (:chunk/group descriptor) (:chunk/start descriptor)
                                (:window-floors (page-pool/route pool id))]))]
          (let [result (manager/restore-paged-prefix!
                        cache pool :restored fingerprint tokens)]
            (is (= 20 (:cached-token-count result)))
            (is (= {:window-8 12}
                   (:window-floors (page-pool/route pool :restored))))
            (is (= {:window-8 12} (nth (first @restored) 3))
                "the route carries its floors before the first row lands")
            (is (= (get-in (page-pool/route pool :restored) [:window-floors :window-8])
                   (apply min (keep (fn [[_ group start]] (when (= :window-8 group) start))
                                    @restored)))
                "the floor is the first window row the restore wrote")
            (is (= #{[:restored :global 0] [:restored :global 4] [:restored :global 8]
                     [:restored :global 12] [:restored :global 16]
                     [:restored :window-8 12] [:restored :window-8 16]}
                   (set (map #(subvec % 0 3) @restored))))))))))
