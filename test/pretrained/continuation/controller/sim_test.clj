(ns pretrained.continuation.controller.sim-test
  (:require [clojure.test :refer [deftest is]]
            [pretrained.continuation.controller.sim :as sim]))

(def request
  #:request{:id :request-1
            :model-fingerprint "gemma/test"
            :tokens (vec (range 33))
            :max-new-tokens 4
            :simulated-output [41 42 43 44]})

(defn- candidate
  [worker tier cached queue load restore prefill]
  #:candidate{:worker-id worker
              :worker-epoch 0
              :cache-tier tier
              :cached-token-count cached
              :cached-bytes (* cached 128)
              :queue-ms queue
              :prefix-load-ms load
              :gpu-restore-ms restore
              :prefill-ms-per-token prefill
              :first-token-ms 1
              :page-size 8
              :free-pages 8
              :evictable-pages 0
              :max-context 128})

(def workers
  {:near #:worker{:epoch 0 :models #{"gemma/test"}
                  :free-pages 8 :evictable-pages 0}
   :far #:worker{:epoch 0 :models #{"gemma/test"}
                 :free-pages 8 :evictable-pages 0}})

(deftest virtual-cluster-routes-restores-prefills-and-decodes
  (let [near (candidate :near :gpu 24 1 0 0 0.5)
        far (candidate :far :object 32 0 30 4 0.5)
        result (-> (sim/make-sim workers)
                   (sim/submit request [far near])
                   (sim/run-until-response :request-1 100))]
    (is (= {:request/id :request-1
            :response/type :completed
            :response/value {:status :completed :tokens [41 42 43 44]}}
           (sim/response result :request-1)))
    (is (= :near
           (get-in result [:sim/router :router/requests :request-1
                           :assignment/candidate :candidate/worker-id])))
    (is (= :completed
           (get-in result [:sim/workers :near :worker/assignments
                           [:request-1 1] :assignment/phase])))))

(deftest actual-worker-capacity-overrides-a-stale-router-snapshot
  (let [stale (candidate :near :gpu 32 0 0 0 0.5)
        fallback (candidate :far :gpu 32 2 0 0 0.5)
        constrained (assoc-in workers [:near :worker/free-pages] 0)
        result (-> (sim/make-sim constrained)
                   (sim/submit request [stale fallback])
                   (sim/run-until-response :request-1 100))]
    (is (= :completed (:response/type (sim/response result :request-1))))
    (is (= :far
           (get-in result [:sim/router :router/requests :request-1
                           :assignment/candidate :candidate/worker-id])))
    (is (some #(= :insufficient-pages
                  (get-in % [:effects 0 :event/reason]))
              (:sim/trace result)))))

(deftest offer-timeout-falls-back-across-a-directed-partition
  (let [first (candidate :near :gpu 32 0 0 0 0.5)
        second (candidate :far :gpu 16 1 0 0 0.5)
        result (-> (sim/make-sim workers {:offer-timeout 3})
                   (sim/partition-link :router :near)
                   (sim/submit request [first second])
                   (sim/run-until-response :request-1 100))]
    (is (= :completed (:response/type (sim/response result :request-1))))
    (is (= [:request-1 2]
           (get-in result [:sim/router :router/requests :request-1
                           :assignment/id])))
    (is (some #(= :network/drop (:trace/type %)) (:sim/trace result)))))

(deftest worker-crash-falls-back-and-late-completion-is-fenced
  (let [slow (candidate :near :ssd 32 0 3 1 0.5)
        fallback (candidate :far :gpu 16 1 0 0 0.5)
        accepted (-> (sim/make-sim workers)
                     (sim/submit request [slow fallback])
                     (sim/run 2))
        crashed (sim/crash accepted :near)
        result (sim/run-until-response crashed :request-1 100)]
    (is (= :completed (:response/type (sim/response result :request-1))))
    (is (= [:request-1 2]
           (get-in result [:sim/router :router/requests :request-1
                           :assignment/id])))
    (is (= 1 (get-in result [:sim/workers :near :worker/epoch])))
    (is (empty? (get-in result [:sim/workers :near :worker/assignments])))))

(deftest simulations-are-replayable-values
  (let [scenario #(-> (sim/make-sim workers {:network-delay 2})
                      (sim/submit request
                                  [(candidate :near :ssd 24 1 3 1 0.5)
                                   (candidate :far :gpu 8 2 0 0 0.5)])
                      (sim/run-until-response :request-1 100))
        left (scenario)
        right (scenario)]
    (is (= left right))
    (is (= (:sim/trace left) (:sim/trace right)))))
