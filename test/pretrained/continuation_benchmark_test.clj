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

(deftest paged-continuation-benchmark-records-restore-ttft-and-context-steps
  (let [resident (atom {})
        transfers (atom {:counters {}})
        completed (fn [value] (CompletableFuture/completedFuture value))
        record-transfer!
        (fn [direction]
          (swap! transfers update-in
                 [:counters [direction :device-event true]]
                 #(merge-with +
                              (or % {})
                              {:submissions 1 :bytes 80 :commands 2
                               :elapsed-ns 20 :submit-host-ns 3
                               :host-wall-ns 25})))]
    (with-redefs [paged-decoder/prime-prompt!
                  (fn [_ continuation-id tokens]
                    (swap! resident update continuation-id
                           #(or % {:continuation-id continuation-id
                                   :token-count (dec (count tokens))})))
                  paged-decoder/step!
                  (fn [_ continuation-id position]
                    (swap! resident update-in [continuation-id :token-count] inc)
                    (+ 100 position))
                  page-pool/route
                  (fn [_ continuation-id]
                    (get @resident continuation-id))
                  page-pool/release-route!
                  (fn [_ continuation-id]
                    (let [present? (contains? @resident continuation-id)]
                      (swap! resident dissoc continuation-id)
                      present?))
                  page-pool/page-pool? (constantly true)
                  page-pool/transfer-stats (fn [_] @transfers)
                  manager/checkpoint-paged-chunks-async!
                  (fn [& _]
                    (record-transfer! :download)
                    {:accepted? true
                     :captured (completed [{:bytes 80}])
                     :published (completed ::published)})
                  manager/restore-paged-prefix!
                  (fn [_ _ continuation-id _ tokens]
                    (record-transfer! :upload)
                    (let [cached (dec (count tokens))]
                      (swap! resident assoc continuation-id
                             {:continuation-id continuation-id
                              :token-count cached})
                      {:cached-token-count cached}))
                  manager/stats (fn [_] {:full-hits 4})]
      (let [result
            (benchmark/benchmark-paged-continuation!
             ::cache {:pool ::pool :decode-state {:batch-size 1}}
             "fixture-v1" [1 2 3 4]
             {:iterations 2 :warmups 1 :decode-tokens 3})
            first-restored (get-in result [:restored :first-measured])]
        (is (= 80 (get-in result [:checkpoint :stored-bytes])))
        (is (= 3 (:cached-token-count first-restored)))
        (is (= [4 5 6]
               (mapv :context-token-count (:steps first-restored))))
        (is (= [103 104 105]
               (mapv :token (:steps first-restored))))
        (is (= 6 (count (get-in result [:restored :warm :decode :samples-ms]))))
        (is (pos? (get-in result [:restored :warm :decode :tokens-per-second])))
        (is (= 1 (get-in result
                         [:checkpoint :transfer :counters
                          [:download :device-event true] :submissions])))
        (is (= 2 (get-in result
                         [:restored :warm :prefix-transfer :totals :counters
                          [:upload :device-event true] :submissions])))
        (is (= {:full-hits 4} (:cache-stats result)))
        (is (empty? @resident)
            "benchmark releases warmup, source, and measured routes")))))
