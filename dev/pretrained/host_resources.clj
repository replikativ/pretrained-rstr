(ns pretrained.host-resources
  "Linux host resource facts and admission used by the real-model smokes."
  (:require [clojure.string :as str])
  (:import (java.nio.file Files Path)))

(def ^:private gib (* 1024 1024 1024))

(defn- read-lines
  [path]
  ;; GraalVM's BufferedInputStream calls FileInputStream.available(), which
  ;; returns EINVAL for procfs. The NIO channel reader does not use available().
  (vec (Files/readAllLines (Path/of path (make-array String 0)))))

(defn resource-snapshot
  "Return the Linux resource facts used to guard real-model smokes."
  []
  (let [meminfo
        (into {}
              (keep (fn [line]
                      (when-let [[_ field kib]
                                 (re-matches #"([^:]+):\s+(\d+)\s+kB" line)]
                        [(keyword field) (* 1024 (Long/parseLong kib))])))
              (read-lines "/proc/meminfo"))
        load-one (Double/parseDouble
                  (first (str/split (first (read-lines "/proc/loadavg")) #"\s+")))]
    {:available-bytes (:MemAvailable meminfo)
     :swap-total-bytes (:SwapTotal meminfo)
     :swap-free-bytes (:SwapFree meminfo)
     :load-one load-one
     :processors (.availableProcessors (Runtime/getRuntime))}))

(defn resource-admission
  "Explain whether `snapshot` is safe for a real-model compile.

  Thresholds default to 14 GiB available memory, 256 MiB free swap, and a
  one-minute load of 1.25 per processor."
  ([snapshot] (resource-admission snapshot {}))
  ([{:keys [available-bytes swap-total-bytes swap-free-bytes load-one processors]
     :as snapshot}
    {:keys [minimum-available-bytes minimum-swap-free-bytes
            maximum-load-per-processor]
     :or {minimum-available-bytes (* 14 gib)
          minimum-swap-free-bytes (* 256 1024 1024)
          maximum-load-per-processor 1.25}}]
   (let [load-per-processor (/ (double load-one) (max 1 (long processors)))
         reasons
         (cond-> []
           (< (long available-bytes) (long minimum-available-bytes))
           (conj :insufficient-available-memory)
           (and (pos? (long swap-total-bytes))
                (< (long swap-free-bytes) (long minimum-swap-free-bytes)))
           (conj :insufficient-free-swap)
           (> load-per-processor (double maximum-load-per-processor))
           (conj :host-load-too-high))]
     {:admitted? (empty? reasons)
      :reasons reasons
      :load-per-processor load-per-processor
      :snapshot snapshot})))

(defn preflight
  "Return current resource admission without loading weights or opening a GPU."
  ([] (preflight {}))
  ([thresholds] (resource-admission (resource-snapshot) thresholds)))
