(ns pretrained.continuation.controller.protocol-test
  (:require [clojure.test :refer [deftest is]]
            [pretrained.continuation.controller.protocol :as protocol]))

(deftest generation-requests-are-normalized-at-the-boundary
  (is (= #:request{:id :r1
                   :model-fingerprint "gemma/test"
                   :tokens [1 2 3]
                   :max-new-tokens 4
                   :arrival 0
                   :priority 0}
         (protocol/generation-request
          #:request{:id :r1
                    :model-fingerprint "gemma/test"
                    :tokens '(1 2 3)
                    :max-new-tokens 4})))
  (doseq [request [#:request{:model-fingerprint "m" :tokens [1]
                             :max-new-tokens 1}
                   #:request{:id :r :model-fingerprint "" :tokens [1]
                             :max-new-tokens 1}
                   #:request{:id :r :model-fingerprint "m" :tokens []
                             :max-new-tokens 1}
                   #:request{:id :r :model-fingerprint "m" :tokens [1]
                             :max-new-tokens 0}]]
    (is (thrown? clojure.lang.ExceptionInfo
                 (protocol/generation-request request)))))

(deftest approximate-cache-candidates-are-refused
  (is (thrown-with-msg?
       clojure.lang.ExceptionInfo #"Approximate"
       (protocol/worker-candidate
        #:candidate{:worker-id :w
                    :worker-epoch 0
                    :cache-tier :peer
                    :cached-token-count 1
                    :cached-bytes 1
                    :queue-ms 0
                    :prefix-load-ms 1
                    :gpu-restore-ms 1
                    :prefill-ms-per-token 1
                    :first-token-ms 1
                    :page-size 16
                    :free-pages 1
                    :evictable-pages 0
                    :max-context 32
                    :exact? false}))))
