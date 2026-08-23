(ns pretrained.continuation-gpu-test
  (:require [clojure.test :refer [deftest is testing]]
            [pretrained.continuation.gpu :as continuation-gpu]
            [pretrained.decoder-gpu :as decoder-gpu]
            [raster.gpu.core :as gpu]))

(def ^:private model
  {:n-layers 2 :n-kv 1 :head-dim 2 :vocab 17})

(def ^:private dstate
  {:model model :maxpos 12 :sess ::session})

(deftest gpu-boundary-keeps-the-pending-token-resident
  (let [decoded (atom []) primed (atom []) stepped (atom [])]
    (with-redefs [decoder-gpu/decode-token!
                  (fn [_ token position] (swap! decoded conj [token position]) (float-array 1))
                  decoder-gpu/prime-resident-token!
                  (fn [state token] (swap! primed conj token) state)
                  decoder-gpu/resident-step!
                  (fn [_ position] (swap! stepped conj position) (+ 10 position))]
      (let [state (continuation-gpu/start-gpu dstate [2 3 4]
                                               {:model-fingerprint "fixture-v1"})
            [next-state token] (continuation-gpu/step-gpu state)]
        (is (= [[2 0] [3 1]] @decoded))
        (is (= [4] @primed))
        (is (= [2] @stepped))
        (is (= 12 token))
        (is (= 3 (:continuation/processed-count next-state)))
        (is (= 12 (:continuation/pending-token next-state)))
        (is (= [2 3 4 12] (:continuation/tokens next-state)))))))

(deftest gpu-export-and-restore-transfer-only-the-occupied-prefix
  (let [downloads (atom nil) uploads (atom nil) primed (atom nil)]
    (with-redefs [decoder-gpu/decode-token! (fn [& _] (float-array 1))
                  decoder-gpu/prime-resident-token!
                  (fn [state token] (reset! primed token) state)
                  gpu/download-ranges!
                  (fn [_ entries]
                    (reset! downloads entries)
                    (doseq [[key ^floats destination _] entries]
                      (java.util.Arrays/fill destination
                                             (float (if (.startsWith (name key) "kc") 1.0 2.0))))
                    (mapv second entries))
                  gpu/upload-ranges!
                  (fn [_ entries] (reset! uploads entries) (mapv second entries))]
      (let [state (assoc (continuation-gpu/start-gpu dstate [2 3 4]
                                                       {:model-fingerprint "fixture-v1"})
                         :continuation/processed-count 5
                         :continuation/pending-token 9
                         :continuation/tokens [2 3 4 5 6 9])
            snapshot (continuation-gpu/export-gpu state)
            restored (continuation-gpu/restore-gpu dstate snapshot
                                                   {:model-fingerprint "fixture-v1"})]
        (testing "two K/V buffers per layer move as one batch"
          (is (= [:kc0 :vc0 :kc1 :vc1] (mapv first @downloads)))
          (is (= [:kc0 :vc0 :kc1 :vc1] (mapv first @uploads))))
        (testing "five occupied rows of two floats each move, not maxpos capacity"
          (is (every? #(= {:elements 10} (nth % 2)) @downloads))
          (is (every? #(= 10 (alength ^floats (second %))) @uploads)))
        (is (= 9 @primed))
        (is (= 5 (:continuation/processed-count restored)))
        (is (= [2 3 4 5 6 9] (:continuation/tokens restored)))))))

(deftest chunk-transfer-uses-host-and-device-offsets
  (let [downloads (atom nil)
        uploads (atom nil)
        decoded (atom [])
        primed (atom nil)
        state {:continuation/backend :gpu
               :continuation/dstate dstate
               :continuation/model-fingerprint "fixture-v1"
               :continuation/layout {:n-layers 2 :n-kv 1 :head-dim 2}
               :continuation/processed-count 6
               :continuation/pending-token 7
               :continuation/tokens [1 2 3 4 5 6 7]}
        descriptor {:chunk/start 2 :chunk/token-count 2
                    :chunk/prefix-hash (random-uuid)}]
    (with-redefs [gpu/download-ranges!
                  (fn [_ entries]
                    (reset! downloads entries)
                    (doseq [[key ^floats destination {:keys [dst-element elements]}] entries]
                      (java.util.Arrays/fill destination (int dst-element)
                                             (int (+ dst-element elements))
                                             (float (case key :kc0 1 :kc1 2 :vc0 3 :vc1 4))))
                    (mapv second entries))
                  gpu/upload-ranges!
                  (fn [_ entries] (reset! uploads entries) (mapv second entries))
                  decoder-gpu/decode-token!
                  (fn [_ token position] (swap! decoded conj [token position]))
                  decoder-gpu/prime-resident-token!
                  (fn [state* token] (reset! primed token) state*)]
      (let [tensor-chunk (continuation-gpu/export-gpu-chunk state descriptor)]
        (is (= [1.0 1.0 1.0 1.0
                2.0 2.0 2.0 2.0
                3.0 3.0 3.0 3.0
                4.0 4.0 4.0 4.0]
               (vec (:chunk/payload tensor-chunk))))
        (is (every? #(= 4 (get-in % [2 :src-element])) @downloads))
        (continuation-gpu/upload-gpu-chunk! dstate descriptor
                                            (:chunk/payload tensor-chunk))
        (is (= [0 8 4 12] (mapv #(get-in % [2 :src-element]) @uploads)))
        (is (every? #(= 4 (get-in % [2 :dst-element])) @uploads)))
      (let [resumed (continuation-gpu/resume-prompt-from-prefix
                     dstate "fixture-v1" [1 2 3 4 5 6] 4)]
        (is (= [[5 4]] @decoded))
        (is (= 6 @primed))
        (is (= 5 (:continuation/processed-count resumed)))))))
