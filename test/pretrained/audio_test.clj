(ns pretrained.audio-test
  "Model-free: WAV decode exactness on a synthesized sine (peak/RMS analytic),
  stereo mixdown, resampling."
  (:require [clojure.test :refer [deftest is testing]]
            [pretrained.audio :as audio])
  (:import [java.io File DataOutputStream FileOutputStream ByteArrayOutputStream]))

(defn- write-wav16!
  "Minimal 16-bit PCM WAV writer: interleaved channels."
  [^File f rate channels ^shorts samples]
  (let [n (alength samples)
        data-bytes (* 2 n)
        bos (ByteArrayOutputStream.)
        le16 (fn [^long v] (.write bos (bit-and v 0xFF)) (.write bos (bit-and (bit-shift-right v 8) 0xFF)))
        le32 (fn [^long v] (le16 (bit-and v 0xFFFF)) (le16 (bit-and (bit-shift-right v 16) 0xFFFF)))]
    (.write bos (.getBytes "RIFF")) (le32 (+ 36 data-bytes))
    (.write bos (.getBytes "WAVE")) (.write bos (.getBytes "fmt "))
    (le32 16) (le16 1) (le16 channels) (le32 rate)
    (le32 (* rate channels 2)) (le16 (* channels 2)) (le16 16)
    (.write bos (.getBytes "data")) (le32 data-bytes)
    (dotimes [i n] (le16 (bit-and (long (aget samples i)) 0xFFFF)))
    (with-open [out (FileOutputStream. f)] (.write out (.toByteArray bos)))))

(deftest sine-exactness
  (testing "440Hz stereo 22050Hz → 16k mono: analytic peak and RMS"
    (let [f (File/createTempFile "sine" ".wav")
          sr 22050 n sr amp 12000
          xs (short-array (* 2 n))]
      (dotimes [i n]
        (let [v (short (Math/round (* amp (Math/sin (/ (* 2.0 Math/PI 440.0 i) sr)))))]
          (aset xs (* 2 i) v) (aset xs (inc (* 2 i)) v)))
      (write-wav16! f sr 2 xs)
      (let [{:keys [^floats samples rate]} (audio/load-wav (.getPath f))
            peak (loop [i 0 m 0.0] (if (< i (alength samples))
                                     (recur (inc i) (max m (Math/abs (double (aget samples i))))) m))
            rms (Math/sqrt (/ (loop [i 0 s 0.0]
                                (if (< i (alength samples))
                                  (recur (inc i) (+ s (let [v (double (aget samples i))] (* v v)))) s))
                              (alength samples)))]
        (is (= 16000 rate))
        (is (< (Math/abs (- peak (/ amp 32768.0))) 0.002) "peak = amp/2^15")
        (is (< (Math/abs (- rms (/ (/ amp 32768.0) (Math/sqrt 2.0)))) 0.002) "RMS = peak/sqrt2")
        (is (= 15999 (alength samples)) "duration preserved through resample"))
      (.delete f))))

(deftest mono-passthrough-no-resample
  (testing "16kHz mono passes through without resample artifacts"
    (let [f (File/createTempFile "mono" ".wav")
          xs (short-array 1600)]
      (dotimes [i 1600] (aset xs i (short (- (mod (* i 37) 2000) 1000))))
      (write-wav16! f 16000 1 xs)
      (let [{:keys [^floats samples]} (audio/load-wav (.getPath f))]
        (is (= 1600 (alength samples)))
        (is (< (Math/abs (- (aget samples 5) (/ (- (mod (* 5 37) 2000) 1000) 32768.0))) 1e-6)))
      (.delete f))))
