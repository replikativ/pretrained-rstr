(ns pretrained.continuation-paged-runtime-test
  (:require [clojure.test :refer [deftest is]]
            [pretrained.continuation.paged-runtime :as paged-runtime])
  (:import (java.util.concurrent CancellationException)))

(defn- effect
  [assignment-id continuation-id tokens max-new]
  {:assignment/id assignment-id
   :assignment/request
   {:request/id assignment-id
    :request/continuation-id continuation-id
    :request/model-fingerprint "fixture-model-v1"
    :request/tokens tokens
    :request/max-new-tokens max-new}})

(defn- await-value
  [timeout-ms operation]
  (let [deadline (+ (System/nanoTime) (* 1000000 timeout-ms))]
    (loop []
      (if-let [value (operation)]
        value
        (if (< (System/nanoTime) deadline)
          (do (Thread/sleep 5) (recur))
          (throw (ex-info "Timed out waiting for paged runtime state"
                          {:timeout-ms timeout-ms})))))))

(deftest prefill-and-decode-share-sparse-fixed-lanes
  (let [route-counts (atom {:prefill 0 :decode 2})
        primes (atom [])
        submissions (atom [])
        first-step (promise)
        release-first (promise)
        calls (atom 0)
        runtime
        (paged-runtime/open-runtime
         {:decode-state {:batch-size 3 :maxpos 64}}
         {:prime-lanes! (fn [_ values] (swap! primes conj values))
          :step-lanes!
          (fn [_ work]
            (swap! submissions conj work)
            (when (= 1 (swap! calls inc))
              (deliver first-step true)
              @release-first)
            (mapv (fn [{:keys [lane continuation-id position] :as item}]
                    (swap! route-counts update continuation-id inc)
                    (assoc item :lane lane :token (+ 100 position)))
                  work))
          :route-token-count (fn [_ id] (get @route-counts id))})
        prefill-result
        (future
          (paged-runtime/prefill!
           runtime (effect :prefill-assignment :prefill [1 2 3 4 5] 2)))]
    (try
      (is (true? (deref first-step 1000 false)))
      (let [decode-result
            (future
              (paged-runtime/decode!
               runtime (effect :decode-assignment :decode [7 8 9] 3)))]
        (deliver release-first true)
        (is (= {:ok? true} (deref prefill-result 1000 ::timeout)))
        (is (= {:ok? true :tokens [102 103 104] :stop-reason :length}
               (deref decode-result 1000 ::timeout)))
        (is (some #(= #{:prefill :decode}
                      (set (map :continuation-id %)))
                  @submissions))
        (is (some #(and (= 2 (count %))
                        (contains? (set (map :token %)) 9))
                  @primes)
            "a prompt token and refilled decode token are primed together")
        (is (= 4 (:prefill @route-counts)))
        (is (= 5 (:decode @route-counts)))
        (is (>= (:scheduled-tokens (paged-runtime/state runtime)) 7)))
      (finally
        (deliver release-first true)
        (.close runtime)))))

(deftest interrupted-handler-removes-queued-generation-after-current-step
  (let [route-counts (atom {:decode 2})
        entered (promise)
        release-step (promise)
        runtime
        (paged-runtime/open-runtime
         {:decode-state {:batch-size 1 :maxpos 64}}
         {:prime-lanes! (fn [& _] nil)
          :step-lanes!
          (fn [_ work]
            (deliver entered true)
            @release-step
            (mapv (fn [{:keys [continuation-id position] :as item}]
                    (swap! route-counts update continuation-id inc)
                    (assoc item :token (+ 100 position)))
                  work))
          :route-token-count (fn [_ id] (get @route-counts id))})
        task (future
               (try
                 (paged-runtime/decode!
                  runtime (effect :decode-assignment :decode [7 8 9] 20))
                 (catch CancellationException _
                   :cancelled)))]
    (try
      (is (true? (deref entered 1000 false)))
      (is (= 1 (paged-runtime/cancel-assignment!
                runtime :decode-assignment)))
      (deliver release-step true)
      (is (= :cancelled (deref task 1000 ::timeout)))
      (await-value
       1000
       #(when (and (>= (:decode @route-counts) 3)
                   (every? nil? (:lanes (paged-runtime/state runtime))))
          true))
      (let [count-after-cancel (:decode @route-counts)]
        (Thread/sleep 30)
        (is (= count-after-cancel (:decode @route-counts)))
        (is (<= count-after-cancel 3)))
      (finally
        (deliver release-step true)
        (.close runtime)))))

(deftest background-restore-does-not-block-unrelated-decode-lanes
  (let [route-counts (atom {:decode 2})
        entered (promise)
        release-restore (promise)
        runtime
        (paged-runtime/open-runtime
         {:decode-state {:batch-size 1 :maxpos 64}}
         {:prime-lanes! (fn [& _] nil)
          :step-lanes!
          (fn [_ work]
            (mapv (fn [{:keys [continuation-id position] :as item}]
                    (swap! route-counts update continuation-id inc)
                    (assoc item :token (+ 100 position)))
                  work))
          :route-token-count (fn [_ id] (get @route-counts id))})
        restore
        (future
          (paged-runtime/run-background-operation!
           runtime :restore-assignment
           (fn [_cancelled?]
             (deliver entered true)
             @release-restore
             {:ok? true})))]
    (try
      (is (true? (deref entered 1000 false)))
      (is (= {:ok? true :tokens [102 103] :stop-reason :length}
             (deref
              (future
                (paged-runtime/decode!
                 runtime (effect :decode-assignment :decode [7 8 9] 2)))
              1000 ::timeout)))
      (is (not (realized? restore))
          "decode completes while the independent restore boundary is pending")
      (deliver release-restore true)
      (is (= {:ok? true} (deref restore 1000 ::timeout)))
      (finally
        (deliver release-restore true)
        (.close runtime)))))

(deftest decode-rejects-an-incomplete-prompt-without-poisoning-the-runtime
  (let [runtime
        (paged-runtime/open-runtime
         {:decode-state {:batch-size 2 :maxpos 64}}
         {:prime-lanes! (fn [& _] nil)
          :step-lanes! (fn [& _] [])
          :route-token-count (fn [& _] 1)})]
    (try
      (is (thrown-with-msg?
           clojure.lang.ExceptionInfo
           #"completely primed"
           (paged-runtime/decode!
            runtime (effect :decode-assignment :decode [7 8 9] 2))))
      (is (zero? (:iterations (paged-runtime/state runtime))))
      (finally
        (.close runtime)))))

(deftest single-lane-prefill-uses-exact-bulk-tiles-and-a-scalar-tail
  (let [route-counts (atom {:prefill 0})
        bulk-calls (atom [])
        scalar-calls (atom [])
        runtime
        (paged-runtime/open-runtime
         {:decode-state {:batch-size 1 :maxpos 64}}
         {:prefill-tile-size 3
          :prefill-range!
          (fn [_ continuation-id tokens position]
            (swap! bulk-calls conj [continuation-id tokens position])
            (swap! route-counts update continuation-id + (count tokens)))
          :prime-lanes! (fn [& _] nil)
          :step-lanes!
          (fn [_ work]
            (mapv (fn [{:keys [continuation-id position] :as item}]
                    (swap! scalar-calls conj [continuation-id position])
                    (swap! route-counts update continuation-id inc)
                    (assoc item :token (+ 100 position)))
                  work))
          :route-token-count (fn [_ id] (get @route-counts id))})]
    (try
      (is (= {:ok? true}
             (paged-runtime/prefill!
              runtime (effect :prefill-assignment :prefill
                              [10 11 12 13 14 15 16 17] 1))))
      (is (= [[:prefill [10 11 12] 0]
              [:prefill [13 14 15] 3]]
             @bulk-calls))
      (is (= [[:prefill 6]] @scalar-calls))
      (is (= 7 (:prefill @route-counts)))
      (let [state (paged-runtime/state runtime)]
        (is (= 2 (:bulk-prefill-tiles state)))
        (is (= 6 (:bulk-prefill-tokens state)))
        (is (= 3 (:prefill-tile-size state)))
        (is (number? (:bulk-prefill-ms-per-token state)))
        (is (= 3 (get-in state
                         [:calibration :prefill-ms-per-token :count])))
        (is (number? (get-in state
                             [:calibration :prefill-ms-per-token :ewma])))
        (is (= 7 (:scheduled-tokens state))))
      (finally
        (.close runtime)))))

(deftest decode-preempts-prefill-at-the-next-bulk-tile-boundary
  (let [route-counts (atom {:prefill 0 :decode 2})
        calls (atom [])
        entered-first-tile (promise)
        release-first-tile (promise)
        runtime
        (paged-runtime/open-runtime
         {:decode-state {:batch-size 1 :maxpos 64}}
         {:prefill-tile-size 3
          :prefill-range!
          (fn [_ continuation-id tokens position]
            (swap! calls conj [:bulk continuation-id position])
            (when (zero? position)
              (deliver entered-first-tile true)
              @release-first-tile)
            (swap! route-counts update continuation-id + (count tokens)))
          :prime-lanes! (fn [& _] nil)
          :step-lanes!
          (fn [_ work]
            (mapv (fn [{:keys [continuation-id position] :as item}]
                    (swap! calls conj [:step continuation-id position])
                    (swap! route-counts update continuation-id inc)
                    (assoc item :token (+ 100 position)))
                  work))
          :route-token-count (fn [_ id] (get @route-counts id))})
        prefill-result
        (future
          (paged-runtime/prefill!
           runtime (effect :prefill-assignment :prefill
                           [10 11 12 13 14 15 16 17] 1)))]
    (try
      (is (true? (deref entered-first-tile 1000 false)))
      (let [decode-result
            (future
              (paged-runtime/decode!
               runtime (effect :decode-assignment :decode [7 8 9] 1)))]
        (await-value
         1000
         #(when (= 2 (:active-job-count (paged-runtime/state runtime))) true))
        (deliver release-first-tile true)
        (is (= {:ok? true :tokens [102] :stop-reason :length}
               (deref decode-result 1000 ::timeout)))
        (is (= {:ok? true} (deref prefill-result 1000 ::timeout)))
        (is (= [[:bulk :prefill 0]
                [:step :decode 2]
                [:bulk :prefill 3]
                [:step :prefill 6]]
               @calls)))
      (finally
        (deliver release-first-tile true)
        (.close runtime)))))
