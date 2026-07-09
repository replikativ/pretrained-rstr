(ns pretrained.asr
  "Unified speech-to-text API, mirroring pretrained.embed.

    (def m (asr/load-asr :moonshine-streaming-medium dir))
    (asr/transcribe m \"voice.ogg\")   ;; any format ffmpeg reads (Telegram etc.)

  Engines: :moonshine (245M MIT, English, true streaming — see
  pretrained.asr.moonshine/stream-push! for live use) and :qwen3-asr
  (0.6B/1.7B Apache, 52 languages, :language / :context opts)."
  (:require [pretrained.audio :as audio]))

(def registry
  {:moonshine-streaming-medium
   {:hf "UsefulSensors/moonshine-streaming-medium" :engine :moonshine
    :langs #{:en} :streaming? true}
   :qwen3-asr-0.6b
   {:hf "Qwen/Qwen3-ASR-0.6B-hf" :engine :qwen3-asr :langs :multilingual}
   :qwen3-asr-1.7b
   {:hf "Qwen/Qwen3-ASR-1.7B-hf" :engine :qwen3-asr :langs :multilingual}})

(defn load-asr
  "Load an ASR model by registry key — weights auto-download from HF into the
  local cache on first use (pretrained.hub). Pass an explicit dir to skip."
  ([k]
   (let [entry (get registry k)]
     (assert entry (str "unknown ASR model " k " — known: " (keys registry)))
     (load-asr k ((requiring-resolve 'pretrained.hub/ensure-model) (:hf entry)))))
  ([k dir]
  (let [entry (get registry k)]
    (assert entry (str "unknown ASR model " k " — known: " (keys registry)))
    (case (:engine entry)
      :moonshine (do (require 'pretrained.asr.moonshine)
                     {::entry entry
                      ::model ((requiring-resolve 'pretrained.asr.moonshine/load-model) dir)
                      ::f (requiring-resolve 'pretrained.asr.moonshine/transcribe)})
      :qwen3-asr (do (require 'pretrained.asr.qwen3-asr)
                     {::entry entry
                      ::model ((requiring-resolve 'pretrained.asr.qwen3-asr/load-model) dir)
                      ::f (requiring-resolve 'pretrained.asr.qwen3-asr/transcribe)})))))

(defn transcribe
  "Audio file path (wav/mp3/ogg/opus/m4a/flac — non-WAV needs ffmpeg on PATH) or
  {:samples float[] :rate 16000} → transcription string. opts (qwen3-asr):
  :language e.g. \"English\", :context hotword/bias text."
  ([m wav] (transcribe m wav {}))
  ([m wav opts]
   (let [audio* (if (string? wav) (audio/load-audio wav) wav)]
     (cond
       (= :qwen3-asr (get-in m [::entry :engine]))
       ((::f m) (::model m) audio* opts)
       ;; word timestamps: cross-attention DTW alignment (moonshine)
       (:timestamps? opts)
       ((requiring-resolve 'pretrained.asr.moonshine/transcribe-ts) (::model m) audio*)
       :else
       ((::f m) (::model m) audio*)))))
