(ns pretrained.continuation-paged-decoder-test
  (:require [clojure.string :as str]
            [clojure.test :refer [deftest is testing]]
            [pretrained.attention-state :as attention-state]
            [pretrained.continuation.page-pool :as page-pool]
            [pretrained.continuation.paged-decoder :as paged-decoder]
            [raster.compiler.ir.kernel-abi :as kernel-abi]
            [raster.compiler.ir.kernel-artifact :as kernel-artifact]
            [raster.compiler.ir.kernel-launch :as kernel-launch]
            [raster.compiler.ir.link-plan :as link]
            [raster.gpu.core :as gpu]
            [raster.gpu.link :as gpu-link]))

(def ^:private model
  {:n-layers 2 :n-q 2 :n-kv 1 :head-dim 4 :maxpos 8})

(defn- resident-step
  [phase accesses]
  (let [kernel-name (str/replace (name phase) "-" "_")
        abi (mapv (fn [[symbol dtype access]]
                    (kernel-abi/slot
                     symbol (if (= :read access) :input :output) dtype
                     :role (when (= :read-write access) :inout)))
                  accesses)
        artifact
        (kernel-artifact/make
         {:kernel-name kernel-name
          :source
          (str "__kernel void " kernel-name "("
               (str/join
                ", "
                (map (fn [[symbol dtype access]]
                       (str "__global " (when (= :read access) "const ")
                            (case dtype :float "float" :int "int")
                            "* " (name symbol)))
                     accesses))
               ") {}")
          :abi abi
          :arguments (mapv first accesses)
          :launch (kernel-launch/spec {:workgroup-size [1] :group-count [1]})
          :effects {:kind :map
                    :reads (into #{} (keep (fn [[symbol _ access]]
                                             (when (contains? #{:read :read-write} access)
                                               symbol)))
                                 accesses)
                    :writes (into #{} (keep (fn [[symbol _ access]]
                                              (when (contains? #{:write :read-write} access)
                                                symbol)))
                                  accesses)}})]
    {:phase phase
     :kernel-name kernel-name
     :convention :map
     :artifact artifact
     :argument-specs
     (mapv (fn [[symbol _ access]]
             {:kind (if (= :read access) :input :output) :sym symbol})
           accesses)}))

(defn- staged-executable
  [id accesses roles outputs]
  (let [descriptor {:dtype nil
                    :all-params (mapv first accesses)
                    :array-params (mapv first accesses)
                    :scalar-params []
                    :array-roles {}
                    :allocs []
                    :steps [(resident-step id accesses)]
                    :result-sym (first outputs)}
        bindings (into {} (map (fn [[symbol]] [symbol (keyword (name symbol))])) accesses)
        nodes
        (into {}
              (map (fn [[symbol dtype]]
                     (let [node-id (get bindings symbol)]
                       [node-id
                        (link/node {:id node-id :dtype dtype :shape [8]
                                    :device :ocl:0 :role (get roles node-id :internal)
                                    :ownership :external :allocation-id node-id})])))
              accesses)
        plan (link/make
              {:id id :target :ocl:0 :nodes nodes
               :instances [(link/instance {:id id :descriptor descriptor
                                            :bindings bindings})]
               :outputs outputs})]
    (gpu-link/map->LinkedExecutable {:plan plan})))

(defn- fixture
  []
  (let [pool (page-pool/->DevicePagePool
              ::session (attention-state/layout model) 4 4 :half {}
              (atom {:free (apply sorted-set (range 4))
                     :refcounts {}
                     :leases {}
                     :routes {}}))]
    {:pool pool
     :decoder
     (paged-decoder/map->PagedDecoder
      {:decode-state {:sess ::session :model model :maxpos 8}
       :pool pool
       :executable :paged-executable
       :descriptor-keys {:slots :slots
                         :row-offsets :row-offsets
                         :positions :positions
                         :page-table :page-table
                         :lengths :lengths
                         :start-positions :start-positions}
       :pages-per-sequence 2
       :state (atom {:closed? false})})}))

(deftest routed-graphs-compose-between-generated-stages-before-gpu-allocation
  (let [one-layer-model {:n-layers 1 :n-q 2 :n-kv 1 :head-dim 4
                         :attn-scale 0.5 :maxpos 8}
        pool (page-pool/->DevicePagePool
              ::session (attention-state/layout one-layer-model) 4 4 :half
              {[:key 0] :k0, [:value 0] :v0}
              (atom {:free (sorted-set 0 1 2 3)
                     :refcounts {} :leases {} :routes {}}))
        pre (staged-executable
             :pre
             [['r0 :float :read] ['qr :float :write]
              ['kr :float :write] ['v :float :write]]
             {:r0 :input} [:qr :kr :v])
        post (staged-executable
              :post
              [['r0 :float :read] ['at :float :read] ['r1 :float :write]]
              {:r0 :input :at :input} [:r1])
        head (staged-executable
              :head [['r1 :float :read] ['tokbuf :int :write]]
              {:r1 :input} [:tokbuf])
        decode-state {:sess ::session
                      :model one-layer-model
                      :maxpos 8
                      :device-id :ocl:0
                      :stage-executables
                      {:layers [{:pre pre :post post}]
                       :head-tail head}}
        view #(gpu/->ResidentBufferView ::session % nil)
        captured (atom nil)]
    (with-redefs [gpu/alloc! (fn [& _] nil)
                  gpu/buffer (fn [_ _] (Object.))
                  gpu-link/instantiate!
                  (fn [plan _]
                    (reset! captured plan)
                    ::composite)]
      (let [result
            (#'paged-decoder/linked-paged-executable!
             decode-state pool 2 "fixture"
             (view :qr) (view :kr) (view :v) (view :at))]
        (is (= ::composite (:executable result)))
        (is (= 5 (count (:instances @captured))))
        (is (= [:pre :head]
               [(-> @captured :instances first :id)
                (-> @captured :instances last :id)]))
        (is (= :internal (get-in @captured [:nodes :at :role])))
        (is (= :state (get-in @captured [:nodes :k0 :role])))
        (is (= [:tokbuf] (:outputs @captured)))))))

(deftest staged-step-publishes-only-after-every-layer-completes
  (let [{:keys [pool decoder]} (fixture)
        calls (atom [])]
    (paged-decoder/allocate-continuation! decoder :request)
    (with-redefs [gpu/upload-ranges!
                  (fn [_ entries]
                    (swap! calls conj [:upload (mapv first entries)]))
                  gpu-link/run! (fn [executable]
                                  (swap! calls conj [:run executable]))
                  gpu/download (fn [_ key]
                                 (swap! calls conj [:download key])
                                 (int-array [42]))]
      (is (= 42 (paged-decoder/step! decoder :request 0)))
      (is (= 1 (:token-count (page-pool/route pool :request))))
      (is (= [[:upload [:posbuf :clenbuf :slots :row-offsets :positions
                        :page-table :lengths :start-positions]]
              [:run :paged-executable] [:download :tokbuf]]
             @calls)))))

(deftest failed-layer-leaves-partial-page-writes-unreachable
  (let [{:keys [pool decoder]} (fixture)]
    (paged-decoder/allocate-continuation! decoder :request)
    (with-redefs [gpu/upload-ranges! (fn [& _] nil)
                  gpu-link/run! (fn [_]
                                  (throw (ex-info "linked decode failed" {})))]
      (is (thrown-with-msg? clojure.lang.ExceptionInfo #"linked decode failed"
                            (paged-decoder/step! decoder :request 0)))
      (testing "the logical route never exposes the failed token"
        (let [route (page-pool/route pool :request)]
          (is (zero? (:token-count route)))
          (is (nil? (:pending route))))))))

(deftest positions-must-extend-the-logical-route
  (let [{:keys [decoder]} (fixture)]
    (paged-decoder/allocate-continuation! decoder :request :start-position 9)
    (is (thrown-with-msg? clojure.lang.ExceptionInfo #"does not extend"
                          (paged-decoder/step! decoder :request 8)))))

(deftest restored-prompts-compute-only-the-uncached-suffix
  (let [{:keys [pool decoder]} (fixture)
        calls (atom [])]
    (page-pool/allocate-route! pool :request 2)
    (with-redefs [paged-decoder/decode-token!
                  (fn [_ continuation-id token position]
                    (swap! calls conj [:decode continuation-id token position]))
                  paged-decoder/prime-token!
                  (fn [engine token]
                    (swap! calls conj [:prime token])
                    engine)]
      (is (identical? decoder
                      (paged-decoder/prime-prompt!
                       decoder :request [10 11 12 13 14])))
      (is (= [[:decode :request 12 2]
              [:decode :request 13 3]
              [:prime 14]]
             @calls)))))
