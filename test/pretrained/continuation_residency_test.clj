(ns pretrained.continuation-residency-test
  (:require [clojure.test :refer [deftest is testing]]
            [pretrained.attention-state :as attention-state]
            [pretrained.continuation.page-pool :as page-pool]
            [pretrained.continuation.residency :as residency]))

(def ^:private layout
  (attention-state/layout {:n-layers 1 :n-kv 1 :head-dim 2}))

(defn- fixture-pool
  [state]
  (page-pool/->DevicePagePool
   ::session layout 4 4 :half
   {[:key 0] :pool-k0, [:value 0] :pool-v0}
   (atom state)))

(deftest admission-prefers-low-value-durable-routes
  (let [pool (fixture-pool
              {:free (sorted-set)
               :refcounts {0 1, 1 1, 2 1, 3 1}
               :leases {}
               :routes
               {:cold {:continuation-id :cold :pages [0 1]
                       :token-count 8 :start-position 0
                       :cache/policy {:durable? true :last-access 1
                                      :reuse-probability 0.1
                                      :recompute-ms 10.0 :reload-ms 5.0}}
                :hot {:continuation-id :hot :pages [2 3]
                      :token-count 8 :start-position 0
                      :cache/policy {:durable? true :last-access 2
                                     :reuse-probability 0.9
                                     :recompute-ms 20.0 :reload-ms 2.0}}}})
        result (residency/admit-route!
                pool :incoming 8
                {:policy {:durable? false :last-access 3}})]
    (is (:admissible? result))
    (is (= [:cold] (:evictions result)))
    (is (nil? (page-pool/route pool :cold)))
    (is (= [0 1] (:pages (page-pool/route pool :incoming))))
    (is (= [2 3] (:pages (page-pool/route pool :hot))))))

(deftest admission-never-evicts-pinned-dirty-or-leased-routes
  (let [pool (fixture-pool
              {:free (sorted-set)
               :refcounts {0 1, 1 2, 2 1, 3 1}
               :leases {::lease {:pages [1] :continuation-ids [:leased]}}
               :routes
               {:dirty {:continuation-id :dirty :pages [0]
                        :token-count 4 :start-position 0
                        :cache/policy {:durable? false}}
                :leased {:continuation-id :leased :pages [1]
                         :token-count 4 :start-position 0
                         :cache/policy {:durable? true}}
                :pinned {:continuation-id :pinned :pages [2 3]
                         :token-count 8 :start-position 0
                         :cache/policy {:durable? true :pinned? true}}}})
        before (page-pool/residency-snapshot pool)
        result (residency/admit-route! pool :incoming 4)]
    (is (false? (:admissible? result)))
    (is (= 1 (:shortfall-pages result)))
    (is (= before (page-pool/residency-snapshot pool)))))

(deftest shared-pages-are-accounted-for-as-a-set-of-evictions
  (let [snapshot {:physical-pages 2 :page-size 4 :free-pages #{}
                  :refcounts {0 2, 1 1} :leases {}
                  :routes
                  {:fork-a {:continuation-id :fork-a :pages [0]
                            :cache/policy {:durable? true :last-access 0}}
                   :fork-b {:continuation-id :fork-b :pages [0]
                            :cache/policy {:durable? true :last-access 1}}
                   :other {:continuation-id :other :pages [1]
                           :cache/policy {:durable? true :last-access 2}}}}
        plan (residency/plan-admission snapshot 4)]
    (testing "both low-value sharing routes may be removed before a page is free"
      (is (:admissible? plan))
      (is (= [:fork-a :fork-b] (:evictions plan))))))
