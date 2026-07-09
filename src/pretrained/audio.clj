(ns pretrained.audio
  "Audio input for ASR models: WAV/PCM decode → float[] in [-1,1] at a target
  sample rate (16kHz for Moonshine/Qwen3-ASR/Whisper-family).

  Pure JVM (javax.sound.sampled). Stereo is mixed to mono; sample-rate
  conversion is linear interpolation (fine for speech; the models' conv
  frontends are robust to it — use 16kHz sources when exactness matters)."
  (:import [javax.sound.sampled AudioSystem AudioFormat AudioFormat$Encoding]
           [java.io File ByteArrayOutputStream]))

(defn- bytes->floats
  "Interleaved PCM bytes → mono float[] in [-1,1]."
  ^floats [^bytes b ^AudioFormat fmt]
  (let [ch (.getChannels fmt)
        bits (.getSampleSizeInBits fmt)
        bytes-per (quot bits 8)
        big? (.isBigEndian fmt)
        signed? (= (.getEncoding fmt) AudioFormat$Encoding/PCM_SIGNED)
        frame-bytes (* ch bytes-per)
        n (quot (alength b) frame-bytes)
        out (float-array n)
        scale (float (/ 1.0 (Math/pow 2 (dec bits))))]
    (dotimes [i n]
      (let [acc (loop [c 0 acc 0.0]
                  (if (< c ch)
                    (let [off (+ (* i frame-bytes) (* c bytes-per))
                          raw (case (int bytes-per)
                                1 (let [v (long (aget b off))]
                                    (if signed? v (- (bit-and v 0xFF) 128)))
                                2 (let [b0 (bit-and (long (aget b off)) 0xFF)
                                        b1 (bit-and (long (aget b (inc off))) 0xFF)
                                        v (if big? (bit-or (bit-shift-left b0 8) b1)
                                                   (bit-or (bit-shift-left b1 8) b0))]
                                    (if (>= v 32768) (- v 65536) v))
                                3 (let [b0 (bit-and (long (aget b off)) 0xFF)
                                        b1 (bit-and (long (aget b (inc off))) 0xFF)
                                        b2 (bit-and (long (aget b (+ off 2))) 0xFF)
                                        v (if big?
                                            (bit-or (bit-shift-left b0 16) (bit-shift-left b1 8) b2)
                                            (bit-or (bit-shift-left b2 16) (bit-shift-left b1 8) b0))]
                                    (if (>= v 8388608) (- v 16777216) v)))]
                      (recur (inc c) (+ acc (double raw))))
                    acc))]
        (aset out i (float (* scale (/ acc ch))))))
    out))

(defn- resample-linear
  "Linear-interpolation resample from `from-rate` to `to-rate`."
  ^floats [^floats x from-rate to-rate]
  (if (= (long from-rate) (long to-rate))
    x
    (let [n (alength x)
          ratio (/ (double from-rate) (double to-rate))
          m (long (Math/floor (/ (dec n) ratio)))
          out (float-array m)]
      (dotimes [i m]
        (let [src (* i ratio)
              i0 (long src)
              frac (- src i0)
              a (aget x i0)
              b (aget x (min (inc i0) (dec n)))]
          (aset out i (float (+ (* a (- 1.0 frac)) (* b frac))))))
      out)))

(defn load-wav
  "Read a WAV file → {:samples float[] :rate rate}. Mixes to mono; resamples to
  `target-rate` (default 16000). Handles PCM 8/16/24-bit signed/unsigned."
  ([path] (load-wav path 16000))
  ([path target-rate]
   (with-open [in (AudioSystem/getAudioInputStream (File. ^String path))]
     (let [fmt (.getFormat in)
           _ (assert (contains? #{AudioFormat$Encoding/PCM_SIGNED
                                  AudioFormat$Encoding/PCM_UNSIGNED}
                                (.getEncoding fmt))
                     (str "unsupported encoding " (.getEncoding fmt) " — convert to PCM WAV"))
           bos (ByteArrayOutputStream.)
           buf (byte-array 65536)]
       (loop []
         (let [k (.read in buf)]
           (when (pos? k) (.write bos buf 0 k) (recur))))
       (let [samples (bytes->floats (.toByteArray bos) fmt)
             rate (long (.getSampleRate fmt))]
         {:samples (resample-linear samples rate target-rate)
          :rate (long target-rate)})))))


(defn- load-via-ffmpeg
  "Decode ANY audio format (mp3, ogg/opus — e.g. Telegram voice notes — m4a, flac,
  ...) by shelling out to ffmpeg: raw mono f32le at target-rate on stdout."
  [path target-rate]
  (let [pb (ProcessBuilder.
             ^"[Ljava.lang.String;" (into-array String ["ffmpeg" "-v" "error" "-i" path
                                 "-f" "f32le" "-acodec" "pcm_f32le"
                                 "-ac" "1" "-ar" (str target-rate) "pipe:1"]))
        p (.start pb)
        bos (ByteArrayOutputStream.)
        buf (byte-array 65536)]
    (with-open [in (.getInputStream p)]
      (loop [] (let [k (.read in buf)] (when (pos? k) (.write bos buf 0 k) (recur)))))
    (let [code (.waitFor p)]
      (when-not (zero? code)
        (throw (ex-info (str "ffmpeg failed (" code ") on " path)
                        {:stderr (slurp (.getErrorStream p))}))))
    (let [bytes (.toByteArray bos)
          bb (doto (java.nio.ByteBuffer/wrap bytes)
               (.order java.nio.ByteOrder/LITTLE_ENDIAN))
          n (quot (alength bytes) 4)
          out (float-array n)]
      (.get (.asFloatBuffer bb) out)
      {:samples out :rate (long target-rate)})))

(defn load-audio
  "Read ANY audio file → {:samples float[] :rate}. WAV/PCM decodes pure-JVM
  (load-wav); everything else (mp3, ogg/opus, m4a, flac) via ffmpeg when present."
  ([path] (load-audio path 16000))
  ([path target-rate]
   (if (re-find #"(?i)\.wav$" path)
     (load-wav path target-rate)
     (load-via-ffmpeg path target-rate))))

(defn duration-s
  "Duration in seconds of a loaded audio map."
  ^double [{:keys [^floats samples rate]}]
  (/ (double (alength samples)) (double rate)))
