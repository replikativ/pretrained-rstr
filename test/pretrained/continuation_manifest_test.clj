(ns pretrained.continuation-manifest-test
  (:require [clojure.test :refer [deftest is testing]]
            [datahike.api :as d]
            [pretrained.continuation :as continuation]
            [pretrained.continuation.catalog :as catalog]
            [pretrained.continuation.chunk :as chunk]
            [pretrained.continuation.chunk-store :as chunk-store]
            [pretrained.continuation.manifest :as manifest]
            [pretrained.continuation.manager :as manager]
            [raster.compiler.ir.numerical-state :as numerical-state])
  (:import [java.nio.file Files]
           [java.nio.file.attribute FileAttribute]))

(def ^:private model {:n-layers 2 :n-kv 2 :head-dim 4})
(def ^:private fingerprint "manifest-test/2-layer-gqa-f32-v1")

(defn- values [n offset]
  (let [a (float-array n)]
    (dotimes [i n] (aset a i (float (+ offset (/ i 1000.0)))))
    a))

(defn- fixture [processed]
  (let [elements (* processed (:n-kv model) (:head-dim model))]
    {:continuation/backend :cpu
     :continuation/model-fingerprint fingerprint
     :continuation/layout (continuation/model-layout model)
     :continuation/processed-count processed
     :continuation/pending-token processed
     :continuation/tokens (mapv long (range (inc processed)))
     :continuation/keys (mapv #(values elements (* 10.0 %)) (range 2))
     :continuation/values (mapv #(values elements (+ 100.0 (* 10.0 %))) (range 2))}))

(defn- with-manager [f]
  (let [config {:store {:backend :memory :id (random-uuid)}
                :schema-flexibility :write :keep-history? false :value-caps :default}
        directory (Files/createTempDirectory "manifest-test-" (make-array FileAttribute 0))
        cache (manager/open-manager config directory {:chunk-size 256})]
    (try
      (f cache config)
      (finally
        (.close cache)
        (d/delete-database config)))))

(deftest catalogued-continuation-projects-to-a-certified-manifest
  (with-manager
    (fn [cache _]
      (let [state (fixture 600)
            stored (manager/checkpoint-cpu-chunks! cache state)
            tail (:chunk/prefix-hash (peek stored))
            certified (manifest/continuation-manifest
                       @(:connection cache) fingerprint tail)
            field (first (get-in certified [:manifest :fields]))]
        (is (numerical-state/certified-state? (numerical-state/verify! certified)))
        (is (= (catalog/chunk-entry-id fingerprint tail)
               (get-in certified [:manifest :id]))
            "the manifest id is scoped to the model, not only the tokens")
        (is (= [(catalog/chunk-entry-id fingerprint (:chunk/parent-hash (peek stored)))]
               (get-in certified [:manifest :parents])))
        (is (= [4 600 8] (get-in field [:value :shape])))
        (is (= [4 256 8] (:chunk-shape field)))
        (is (= [[0 0 0] [0 256 0] [0 512 0]] (mapv :offsets (:chunks field))))
        (is (= [4 88 8] (:shape (peek (:chunks field)))))
        (is (= (* 4 600 4 8) (get-in certified [:certificate :logical-byte-length]))
            "logical bytes equal the stored payload bytes")
        (is (= (mapv #(str (:store-key %)) stored)
               (mapv (comp :digest :content) (:chunks field)))
            "content addresses are the chunk store identities")
        (is (= {:mode :fp32-kv :determinism :toleranced :compatibility-id fingerprint}
               (get-in certified [:manifest :numerical-contract])))))))

(deftest every-boundary-reuses-its-parents-content
  (with-manager
    (fn [cache _]
      (let [stored (manager/checkpoint-cpu-chunks! cache (fixture 512))
            [first-chunk second-chunk] stored
            database @(:connection cache)
            at-256 (manifest/continuation-manifest
                    database fingerprint (:chunk/prefix-hash first-chunk))
            at-512 (manifest/continuation-manifest
                    database fingerprint (:chunk/prefix-hash second-chunk))
            addresses #(get-in % [:certificate :content-addresses])]
        (is (= (catalog/chunk-entry-id fingerprint (:chunk/prefix-hash first-chunk))
               (first (get-in at-512 [:manifest :parents]))))
        (is (= (addresses at-256) (subvec (addresses at-512) 0 1))
            "a boundary's manifest shares its parent's chunk addresses")))))

(deftest layout-record-guards-the-fingerprint
  (with-manager
    (fn [cache _]
      (manager/checkpoint-cpu-chunks! cache (fixture 256))
      (let [record (catalog/lookup-layout @(:connection cache) fingerprint)]
        (is (= :fp32-kv (:kv.layout/numerical-mode record)))
        (is (= :float32 (get-in record [:layout :dtype]))))
      (testing "republishing the same layout is idempotent"
        (is (some? (manager/checkpoint-cpu-chunks! cache (fixture 512)))))
      (testing "a second layout under one fingerprint is rejected"
        (let [other (-> (fixture 256)
                        (assoc-in [:continuation/layout :dtype] :float16)
                        (update :continuation/tokens #(mapv (partial + 7) %)))]
          (is (thrown-with-msg?
               clojure.lang.ExceptionInfo #"different durable layout"
               (manager/checkpoint-cpu-chunks! cache other)))))
      (testing "a second numerical contract under one fingerprint is rejected"
        (is (thrown-with-msg?
             clojure.lang.ExceptionInfo #"different numerical contract"
             (catalog/put-chunks!
              (:connection cache) fingerprint
              [(assoc (first (:chunks (chunk/continuation-plan (fixture 256) 256)))
                      :chunk/layout (continuation/model-layout model)
                      :store-key (random-uuid) :bytes 1)]
              {:content-algorithm :hasch/attention-chunk-v3
               :numerical-contract {:mode :fp32-kv :determinism :bitwise}})))))))

(deftest concurrent-first-layouts-cannot-both-commit
  ;; Two publishers that both read "no layout" each carry a swap from nil. The
  ;; second swap fails, and the chunks transacted with it are not published.
  (with-manager
    (fn [cache _]
      (let [connection (:connection cache)
            layout-a (continuation/model-layout model)
            layout-b (assoc layout-a :dtype :float16)
            tx-a (#'catalog/layout-tx! connection fingerprint [{:chunk/layout layout-a}] nil)
            tx-b (#'catalog/layout-tx! connection fingerprint [{:chunk/layout layout-b}] nil)
            node (fn [id] {:kv/id id :kv/kind :kv.kind/chunk
                           :kv/model-fingerprint fingerprint})
            id-a (random-uuid)
            id-b (random-uuid)]
        (is (some #(= :db/cas (first %)) (filter vector? tx-b))
            "both publishers saw no layout")
        (d/transact connection (conj (vec tx-a) (node id-a)))
        (is (thrown? Exception (d/transact connection (conj (vec tx-b) (node id-b)))))
        (is (= :float32 (get-in (catalog/lookup-layout @connection fingerprint)
                                [:layout :dtype])))
        (is (some? (d/entity @connection [:kv/id id-a])))
        (is (nil? (d/entity @connection [:kv/id id-b]))
            "the losing publication's chunks rolled back with its swap")))))

(deftest concurrent-first-publishers-of-one-layout-both-succeed
  ;; The second publisher read "no layout" before the first committed. Its swap
  ;; fails, it re-reads the equal layout, and its chunks publish.
  (with-manager
    (fn [cache _]
      (let [connection (:connection cache)
            layout (continuation/model-layout model)
            descriptors (:chunks (chunk/continuation-plan (fixture 512) 256))
            publish! (fn [descriptor]
                       (catalog/put-chunks!
                        connection fingerprint
                        [(assoc descriptor :chunk/layout layout
                                :store-key (random-uuid) :bytes 1)]
                        {:content-algorithm :hasch/attention-chunk-v3}))
            original catalog/lookup-layout
            stale (atom true)]
        (publish! (first descriptors))
        (with-redefs [catalog/lookup-layout
                      (fn [database fp]
                        (if (compare-and-set! stale true false)
                          nil
                          (original database fp)))]
          (publish! (second descriptors)))
        (is (false? @stale) "the second publisher saw no layout first")
        (let [entry (catalog/lookup-chunk @connection fingerprint
                                          (:chunk/prefix-hash (second descriptors)))]
          (is (some? entry))
          (is (= :hasch/attention-chunk-v3 (:kv/content-algorithm entry))
              "each node records the algorithm that addressed its blobs"))))))

(deftest broken-chains-and-missing-layouts-fail-loudly
  (with-manager
    (fn [cache _]
      (let [database @(:connection cache)]
        (is (thrown-with-msg?
             clojure.lang.ExceptionInfo #"No durable layout"
             (manifest/continuation-manifest database fingerprint (random-uuid)))))
      (manager/checkpoint-cpu-chunks! cache (fixture 256))
      (is (thrown-with-msg?
           clojure.lang.ExceptionInfo #"missing a chunk"
           (manifest/continuation-manifest
            @(:connection cache) fingerprint (random-uuid)))))))

(deftest structured-contract-reaches-the-manifest
  (let [config {:store {:backend :memory :id (random-uuid)}
                :schema-flexibility :write :keep-history? false :value-caps :default}
        directory (Files/createTempDirectory "manifest-test-" (make-array FileAttribute 0))
        cache (manager/open-manager config directory
                                    {:chunk-size 256
                                     :numerical-contract {:mode :fp32-kv-cpu
                                                          :determinism :bitwise}})]
    (try
      (let [stored (manager/checkpoint-cpu-chunks! cache (fixture 256))
            certified (manifest/continuation-manifest
                       @(:connection cache) fingerprint
                       (:chunk/prefix-hash (peek stored)))]
        (is (= {:mode :fp32-kv-cpu :determinism :bitwise :compatibility-id fingerprint}
               (get-in certified [:manifest :numerical-contract]))))
      (finally
        (.close cache)
        (d/delete-database config)))))
