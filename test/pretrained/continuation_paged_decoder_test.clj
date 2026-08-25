(ns pretrained.continuation-paged-decoder-test
  (:require [clojure.test :refer [deftest is testing]]
            [pretrained.attention-state :as attention-state]
            [pretrained.continuation.page-pool :as page-pool]
            [pretrained.continuation.paged-append :as paged-append]
            [pretrained.continuation.paged-attention :as paged-attention]
            [pretrained.continuation.paged-decoder :as paged-decoder]
            [raster.gpu.core :as gpu]))

(def ^:private model
  {:n-layers 2 :n-q 2 :n-kv 1 :head-dim 4 :maxpos 8})

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
     (paged-decoder/->PagedDecoder
      {:sess ::session :model model :maxpos 8}
      pool [:append-0 :append-1] [:attention-0 :attention-1]
      {:layers [{:pre :pre-0 :post :post-0}
                {:pre :pre-1 :post :post-1}]
       :head-tail :head-tail}
      (atom {:closed? false}))}))

(deftest staged-step-publishes-only-after-every-layer-completes
  (let [{:keys [pool decoder]} (fixture)
        calls (atom [])]
    (paged-decoder/allocate-continuation! decoder :request)
    (with-redefs [gpu/upload! (fn [_ key _] (swap! calls conj [:upload key]))
                  gpu/replay! (fn [_ key] (swap! calls conj [:replay key]))
                  gpu/download (fn [_ key]
                                 (swap! calls conj [:download key])
                                 (int-array [42]))
                  paged-append/run! (fn [runner batch]
                                      (swap! calls conj [:append runner])
                                      batch)
                  paged-attention/run!
                  (fn [runner batch]
                    (is (= [:request] (:continuation-ids batch)))
                    (is (= [0] (:positions batch)))
                    (is (= :request
                           (get-in batch [:append-reservations 0 :continuation-id])))
                    (swap! calls conj [:attention runner]))]
      (is (= 42 (paged-decoder/step! decoder :request 0)))
      (is (= 1 (:token-count (page-pool/route pool :request))))
      (is (= [[:upload :posbuf] [:upload :clenbuf]
              [:replay :pre-0] [:append :append-0]
              [:attention :attention-0] [:replay :post-0]
              [:replay :pre-1] [:append :append-1]
              [:attention :attention-1] [:replay :post-1]
              [:replay :head-tail] [:download :tokbuf]]
             @calls)))))

(deftest failed-layer-leaves-partial-page-writes-unreachable
  (let [{:keys [pool decoder]} (fixture)]
    (paged-decoder/allocate-continuation! decoder :request)
    (with-redefs [gpu/upload! (fn [& _] nil)
                  gpu/replay! (fn [_ key]
                                (when (= :post-0 key)
                                  (throw (ex-info "post failed" {}))))
                  paged-append/run! (fn [_ batch] batch)
                  paged-attention/run! (fn [_ _] ::resident-output)]
      (is (thrown-with-msg? clojure.lang.ExceptionInfo #"post failed"
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
