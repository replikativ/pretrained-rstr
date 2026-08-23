(ns pretrained.continuation.benchmark
  "Honest process-warm continuation benchmarks.

  These helpers separate one-time model binding/JIT work, first measured restore,
  warm restore, and asynchronous checkpoint stages. They do not call operating
  system cache-dropping APIs, so `:first-measured-ms` must not be described as a
  cold-SSD measurement."
  (:require [pretrained.continuation :as continuation]
            [pretrained.continuation.gpu :as continuation-gpu]
            [pretrained.continuation.manager :as manager])
  (:import [java.util.concurrent CompletableFuture]))

(defn- timed
  [f]
  (let [started (System/nanoTime)
        value (f)]
    {:value value :milliseconds (/ (- (System/nanoTime) started) 1.0e6)}))

(defn- percentile
  [sorted-samples fraction]
  (let [n (count sorted-samples)
        index (min (dec n) (dec (long (Math/ceil (* fraction n)))))]
    (nth sorted-samples (max 0 index))))

(defn measure!
  "Measure a zero-argument operation after explicit warm-up iterations.

  `opts` accepts positive `:iterations` (default 5) and non-negative `:warmups`
  (default 1). Returns raw millisecond samples plus min/median/p95/max. The
  operation's values are deliberately discarded to keep the result serializable."
  ([operation] (measure! operation {}))
  ([operation {:keys [iterations warmups] :or {iterations 5 warmups 1}}]
   (when-not (and (pos? iterations) (not (neg? warmups)))
     (throw (ex-info "Benchmark iterations must be positive and warmups non-negative"
                     {:iterations iterations :warmups warmups})))
   (dotimes [_ (long warmups)] (operation))
   (let [samples (mapv (fn [_] (:milliseconds (timed operation)))
                       (range (long iterations)))
         sorted-samples (vec (sort samples))]
     {:iterations (long iterations)
      :warmups (long warmups)
      :samples-ms samples
      :min-ms (first sorted-samples)
      :median-ms (percentile sorted-samples 0.5)
      :p95-ms (percentile sorted-samples 0.95)
      :max-ms (peek sorted-samples)})))

(defn- accepted-ticket!
  [ticket]
  (when-not (:accepted? ticket)
    (throw (ex-info "Cache manager rejected the benchmark checkpoint"
                    {:ticket ticket})))
  ticket)

(defn- checkpoint-gpu!
  [cache state]
  (let [submission (timed #(accepted-ticket!
                            (manager/checkpoint-gpu-chunks-async! cache state)))
        ticket (:value submission)
        capture (timed #(.get ^CompletableFuture (:captured ticket)))
        publication (timed #(.get ^CompletableFuture (:published ticket)))]
    {:submission-ms (:milliseconds submission)
     :capture-drain-ms (:milliseconds capture)
     :publication-drain-ms (:milliseconds publication)
     :chunks (count (:value capture))
     :stored-bytes (reduce + 0 (map :bytes (:value capture)))}))

(defn benchmark-gpu-prefix!
  "Benchmark prompt processing against chunked mmap-to-GPU restoration.

  The cache manager and resident `dstate` must already be open/bound; model bind
  time is intentionally excluded. The prompt is checkpointed once, then measured
  as ordinary prefill, first measured restore, and warm restore. `opts` accepts
  `:iterations`, `:warmups`, and optional `:probe-prompt-ids`. A probe typically
  shares some chunks and diverges later, exposing partial-hit behavior.

  The first measurement includes first use within this invocation, but may still
  have compiled queries or cached pages from earlier work in the same JVM. Warm
  restore explicitly includes process and OS page caches.
  Returns timings, speedups, checkpoint drain costs, and manager counters."
  [cache dstate model-fingerprint prompt-ids
   {:keys [iterations warmups probe-prompt-ids]
    :or {iterations 5 warmups 1}}]
  (let [prompt-ids (vec prompt-ids)
        start #(continuation-gpu/start-gpu
                dstate prompt-ids {:model-fingerprint model-fingerprint})
        _ (start)
        prefill (measure! start {:iterations iterations :warmups warmups})
        source (start)
        checkpoint (checkpoint-gpu! cache source)
        restore #(manager/restore-gpu-prefix
                  cache dstate model-fingerprint prompt-ids)
        first-restore (timed restore)
        warm-restore (measure! restore {:iterations iterations :warmups warmups})
        first-ms (:milliseconds first-restore)
        prefill-median (:median-ms prefill)
        warm-median (:median-ms warm-restore)
        probe (when probe-prompt-ids
                (let [result (timed #(manager/restore-gpu-prefix
                                     cache dstate model-fingerprint
                                     (vec probe-prompt-ids)))]
                  {:milliseconds (:milliseconds result)
                   :cached-token-count (get-in result [:value :cached-token-count])
                   :processed-token-count (dec (count probe-prompt-ids))
                   :speedup-vs-prefill (/ prefill-median
                                          (:milliseconds result))}))]
    (cond-> {:prompt {:logical-token-count (count prompt-ids)
                      :processed-token-count (dec (count prompt-ids))}
             :checkpoint checkpoint
             :prefill prefill
             :restore {:first-measured-ms first-ms
                       :warm warm-restore}
             :speedup {:first-measured (/ prefill-median first-ms)
                       :warm (/ prefill-median warm-median)}
             :cache-stats (manager/stats cache)}
      probe (assoc :probe probe))))

(defn benchmark-cpu-snapshot!
  "Benchmark CPU prompt processing against whole-snapshot mmap restoration.

  This archival path materializes JVM tensor arrays after mapping; it is useful
  as a CPU baseline but is distinct from the direct mmap-to-GPU chunk path.
  `opts` requires `:max-position` and accepts `:iterations`/`:warmups`."
  [cache model model-fingerprint prompt-ids
   {:keys [max-position iterations warmups]
    :or {iterations 5 warmups 1}}]
  (let [prompt-ids (vec prompt-ids)
        start #(continuation/start-cpu
                model prompt-ids {:max-position max-position
                                  :model-fingerprint model-fingerprint})
        _ (start)
        prefill (measure! start {:iterations iterations :warmups warmups})
        source (start)
        checkpoint (timed #(manager/checkpoint-cpu! cache source))
        tokens (:continuation/tokens source)
        restore #(let [entry (manager/lookup cache model-fingerprint tokens)]
                   (manager/restore-cpu entry model
                                        {:max-position max-position
                                         :model-fingerprint model-fingerprint}))
        first-restore (timed restore)
        warm-restore (measure! restore {:iterations iterations :warmups warmups})
        prefill-median (:median-ms prefill)
        first-ms (:milliseconds first-restore)
        warm-median (:median-ms warm-restore)]
    {:prompt {:logical-token-count (count prompt-ids)
              :processed-token-count (dec (count prompt-ids))}
     :checkpoint {:milliseconds (:milliseconds checkpoint)
                  :bytes (get-in checkpoint [:value :kv/bytes])}
     :prefill prefill
     :restore {:first-measured-ms first-ms :warm warm-restore}
     :speedup {:first-measured (/ prefill-median first-ms)
               :warm (/ prefill-median warm-median)}}))
