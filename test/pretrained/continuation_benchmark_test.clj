(ns pretrained.continuation-benchmark-test
  (:require [clojure.test :refer [deftest is]]
            [pretrained.continuation.benchmark :as benchmark]
            [pretrained.continuation.gpu :as continuation-gpu]
            [pretrained.continuation.manager :as manager]
            [pretrained.continuation.page-pool :as page-pool]
            [pretrained.continuation.paged-decoder :as paged-decoder])
  (:import [java.util.concurrent CompletableFuture]))

(deftest measure-runs-explicit-warmups-and-samples
  (let [calls (atom 0)
        result (benchmark/measure! #(swap! calls inc)
                                   {:warmups 2 :iterations 3})]
    (is (= 5 @calls))
    (is (= 3 (count (:samples-ms result))))
    (is (<= (:min-ms result) (:median-ms result)
            (:p95-ms result) (:max-ms result)))))

(deftest gpu-benchmark-separates-checkpoint-and-restore-phases
  (let [starts (atom 0)
        restores (atom 0)
        completed (fn [value] (CompletableFuture/completedFuture value))]
    (with-redefs [continuation-gpu/start-gpu
                  (fn [_ prompt opts]
                    (swap! starts inc)
                    {:continuation/tokens prompt
                     :continuation/model-fingerprint (:model-fingerprint opts)})
                  manager/checkpoint-gpu-chunks-async!
                  (fn [_ _]
                    {:accepted? true
                     :captured (completed [{:bytes 64} {:bytes 32}])
                     :published (completed ::published)})
                  manager/restore-gpu-prefix
                  (fn [_ _ _ prompt]
                    (swap! restores inc)
                    {:cached-token-count (dec (count prompt))})
                  manager/stats (fn [_] {:full-hits @restores})]
      (let [result (benchmark/benchmark-gpu-prefix!
                    ::cache ::dstate "fixture-v1" [1 2 3 4]
                    {:iterations 2 :warmups 1
                     :probe-prompt-ids [1 2 9 4]})]
        (is (= 5 @starts))
        (is (= 5 @restores))
        (is (= 96 (get-in result [:checkpoint :stored-bytes])))
        (is (= 3 (get-in result [:probe :cached-token-count])))
        (is (= {:logical-token-count 4 :processed-token-count 3}
               (:prompt result)))))))

(deftest paged-benchmark-measures-restore-through-suffix-readiness
  (let [resident (atom #{})
        primes (atom 0)
        completed (fn [value] (CompletableFuture/completedFuture value))]
    (with-redefs [paged-decoder/prime-prompt!
                  (fn [_ continuation-id _]
                    (swap! resident conj continuation-id)
                    (swap! primes inc))
                  page-pool/route
                  (fn [_ continuation-id]
                    (when (contains? @resident continuation-id)
                      {:continuation-id continuation-id}))
                  page-pool/release-route!
                  (fn [_ continuation-id]
                    (swap! resident disj continuation-id)
                    true)
                  manager/checkpoint-paged-chunks-async!
                  (fn [& _]
                    {:accepted? true
                     :captured (completed [{:bytes 48}])
                     :published (completed ::published)})
                  manager/restore-paged-prefix!
                  (fn [_ _ continuation-id _ tokens]
                    (swap! resident conj continuation-id)
                    {:cached-token-count (if (= [1 2 3 4] (vec tokens)) 3 2)})
                  manager/stats (fn [_] {:full-hits 4 :partial-hits 1})]
      (let [result (benchmark/benchmark-paged-prefix!
                    ::cache {:pool ::pool} "fixture-v1" [1 2 3 4]
                    {:iterations 2 :warmups 1
                     :probe-prompt-ids [1 2 9 4]})]
        (is (= 48 (get-in result [:checkpoint :stored-bytes])))
        (is (= 2 (get-in result [:probe :cached-token-count])))
        (is (= {:full-hits 4 :partial-hits 1} (:cache-stats result)))
        (is (pos? @primes))
        (is (empty? @resident)
            "every benchmark sample releases its worker-local pages")))))
