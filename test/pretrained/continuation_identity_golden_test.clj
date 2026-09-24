(ns pretrained.continuation-identity-golden-test
  "Pin the identities that durable caches depend on.

  Values were computed from the code before attention-state groups existed.
  A model without sliding windows must keep its layout, compatibility
  fingerprint, prefix hash, and chunk content id, or every existing durable
  cache would silently stop matching."
  (:require [clojure.test :refer [deftest is]]
            [pretrained.attention-state :as attention-state]
            [pretrained.continuation :as continuation]
            [pretrained.continuation.chunk :as chunk]
            [pretrained.continuation.chunk-store :as chunk-store]
            [pretrained.model-identity :as model-identity]))

(def ^:private model
  {:n-layers 2 :n-kv 2 :head-dim 4 :arch :fixture :config {:hidden_size 8}
   :desc {:arch :fixture :flags {:rope :single}}})

(def ^:private golden
  {:layout {:version 1 :kind :kv :token-axis :position :dtype :float32
            :byte-order :little-endian
            :slabs [{:name :key :tensor-key :continuation/keys :buffer-prefix "kc"
                     :count 2 :elements-per-token 8}
                    {:name :value :tensor-key :continuation/values :buffer-prefix "vc"
                     :count 2 :elements-per-token 8}]}
   :fingerprint "sha256:e568b4496466f3105dd5dc450d3db6c30d649275aa1a7f84f4e38abb7afcfbea"
   :prefix-hash #uuid "561010d7-575b-5458-ace7-68053cdfdecd"
   :content-id #uuid "364720e5-da2e-598a-b0db-5a504575b9d0"})

(defn- first-chunk [fingerprint]
  (let [n 20
        elements (* n 8)
        values (fn [offset]
                 (let [a (float-array elements)]
                   (dotimes [i elements] (aset a i (float (+ offset (/ i 64.0)))))
                   a))
        state {:continuation/backend :cpu
               :continuation/model-fingerprint fingerprint
               :continuation/layout (continuation/model-layout model)
               :continuation/processed-count n
               :continuation/pending-token n
               :continuation/tokens (mapv long (range (inc n)))
               :continuation/keys [(values 0.0) (values 1.0)]
               :continuation/values [(values 2.0) (values 3.0)]}
        descriptor (first (:chunks (chunk/continuation-plan state 16)))]
    [descriptor (assoc (chunk/cpu-tensor-chunk state descriptor)
                       :chunk/model-fingerprint fingerprint)]))

(deftest windowless-identities-are-stable
  (let [fingerprint (model-identity/compatibility-fingerprint
                     model {:weights-id "golden-weights-v1"
                            :execution-variant :gpu-q4k-paged})
        [descriptor tensor-chunk] (first-chunk fingerprint)]
    (is (= (:layout golden) (attention-state/layout model)))
    (is (= (:fingerprint golden) fingerprint))
    (is (= (:prefix-hash golden) (:chunk/prefix-hash descriptor)))
    (is (= (:content-id golden) (chunk-store/content-id tensor-chunk)))))
