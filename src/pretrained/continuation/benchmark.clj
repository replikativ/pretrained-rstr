(ns pretrained.continuation.benchmark
  "Honest process-warm continuation benchmarks.

  These helpers separate one-time model binding/JIT work, first measured restore,
  warm restore, and asynchronous checkpoint stages. They do not call operating
  system cache-dropping APIs, so `:first-measured-ms` must not be described as a
  cold-SSD measurement."
  (:require [pretrained.continuation :as continuation]
            [pretrained.continuation.gpu :as continuation-gpu]
            [pretrained.continuation.manager :as manager]
            [pretrained.continuation.page-pool :as page-pool]
            [pretrained.continuation.paged-decoder :as paged-decoder])
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

(defn- transfer-snapshot
  [pool]
  (when (page-pool/page-pool? pool)
    (:counters (page-pool/transfer-stats pool))))

(defn- transfer-difference
  [before after]
  (when (and before after)
    {:counters
     (into {}
           (keep
            (fn [counter-key]
              (let [previous (get before counter-key {})
                    current (get after counter-key {})
                    delta (into {}
                                (for [field [:submissions :bytes :commands :elapsed-ns
                                             :submit-host-ns :host-wall-ns]]
                                  [field (- (long (get current field 0))
                                            (long (get previous field 0)))]))]
                (when (pos? (:submissions delta)) [counter-key delta])))
            (set (concat (keys before) (keys after)))))}))

(defn- sum-transfer-differences
  [differences]
  {:counters
   (reduce
    (fn [totals difference]
      (merge-with #(merge-with + %1 %2) totals (:counters difference)))
    {}
    differences)})

(defn- transfer-total
  [transfer field]
  (reduce + 0 (map #(long (get % field 0))
                   (vals (:counters transfer)))))

(declare accepted-ticket!)

(defn- summarize-ms
  [samples]
  (let [samples (mapv double samples)
        sorted-samples (vec (sort samples))]
    {:samples-ms samples
     :min-ms (first sorted-samples)
     :median-ms (percentile sorted-samples 0.5)
     :p95-ms (percentile sorted-samples 0.95)
     :max-ms (peek sorted-samples)}))

(defn- summarize-phase-timings
  [samples]
  (let [phase-maps (vec (keep :restore-phase-timings samples))]
    (when (seq phase-maps)
      (into {}
            (for [phase [:lookup-ms :route-allocation-ms
                         :mapping-lifecycle-ms :gpu-restore-ms :total-ms]
                  :when (every? #(number? (get % phase)) phase-maps)]
              [phase (summarize-ms (mapv phase phase-maps))])))))

(defn- checkpoint-paged!
  [cache pool continuation-id model-fingerprint prompt-ids
   foreground! maximum-foreground-steps transfer-capabilities]
  (let [transfers-before (transfer-snapshot pool)
        checkpoint-started (System/nanoTime)
        submission
        (timed #(accepted-ticket!
                 (manager/checkpoint-paged-chunks-async!
                  cache pool continuation-id model-fingerprint prompt-ids)))
        ticket (:value submission)
        foreground-samples
        (loop [index 0
               samples []]
          (if (and foreground!
                   (< index (long maximum-foreground-steps))
                   (not (.isDone ^CompletableFuture (:captured ticket))))
            (let [sample (timed #(foreground! index))]
              (recur (inc index) (conj samples (:milliseconds sample))))
            samples))
        capture (timed #(.get ^CompletableFuture (:captured ticket)))
        capture-total-ms (/ (- (System/nanoTime) checkpoint-started) 1.0e6)
        publication (timed #(.get ^CompletableFuture (:published ticket)))
        transfer (transfer-difference transfers-before (transfer-snapshot pool))]
    (cond->
     {:submission-ms (:milliseconds submission)
      :capture-drain-ms (:milliseconds capture)
      :capture-total-ms capture-total-ms
      :publication-drain-ms (:milliseconds publication)
      :phase-timings (some-> (:phase-timings ticket) deref)
      :chunks (count (:value capture))
      :stored-bytes (reduce + 0 (map :bytes (:value capture)))}
      foreground!
      (assoc :inference-overlap
             (cond-> {:maximum-steps (long maximum-foreground-steps)
                      :classification
                      (if (:live-overlap-eligible? transfer-capabilities)
                        :eligible
                        :interference-only)
                      :steps-started-before-capture-complete
                      (count foreground-samples)}
               (seq foreground-samples)
               (assoc :step-latency (summarize-ms foreground-samples))))
      transfer (assoc :transfer transfer))))

(defn- continuation-sample!
  [decoder prompt-ids decode-tokens restore!]
  (let [pool (:pool decoder)
        continuation-id (random-uuid)]
    (try
      (let [transfers-before (when restore! (transfer-snapshot pool))
            restore (when restore! (timed #(restore! continuation-id)))
            transfer (when restore!
                       (transfer-difference transfers-before
                                            (transfer-snapshot pool)))
            completion (timed #(paged-decoder/prime-prompt!
                                decoder continuation-id prompt-ids))
            steps
            (loop [index 0
                   position (dec (count prompt-ids))
                   result []]
              (if (< index decode-tokens)
                (let [step (timed #(paged-decoder/step!
                                    decoder continuation-id position))]
                  (recur (inc index) (inc position)
                         (conj result
                               {:index index
                                :position position
                                :context-token-count (inc position)
                                :token (:value step)
                                :milliseconds (:milliseconds step)})))
                result))
            restore-ms (or (:milliseconds restore) 0.0)
            first-token-ms (:milliseconds (first steps))
            decode-ms (reduce + 0.0 (map :milliseconds steps))]
        (cond->
         {:prompt-completion-ms (:milliseconds completion)
          :first-token-ms first-token-ms
          :ready-to-first-token-ms (+ restore-ms
                                      (:milliseconds completion)
                                      first-token-ms)
          :decode-ms decode-ms
          :total-ms (+ restore-ms (:milliseconds completion) decode-ms)
          :steps steps}
          restore
          (assoc :prefix-load-ms restore-ms
                 :cached-token-count
                 (get-in restore [:value :cached-token-count])
                 :restore-phase-timings
                 (get-in restore [:value :restore-phase-timings]))
          transfer
          (assoc :prefix-transfer transfer)))
      (finally
        (when (page-pool/route pool continuation-id)
          (page-pool/release-route! pool continuation-id))))))

(defn- continuation-series!
  [operation iterations warmups]
  (dotimes [_ (long warmups)] (operation))
  (let [samples (mapv (fn [_] (operation)) (range (long iterations)))
        step-times (mapv :milliseconds (mapcat :steps samples))
        decode-summary (summarize-ms step-times)
        transfer-samples (vec (keep :prefix-transfer samples))]
    {:iterations (long iterations)
     :warmups (long warmups)
     :samples samples
     :prompt-completion (summarize-ms (mapv :prompt-completion-ms samples))
     :first-token (summarize-ms (mapv :first-token-ms samples))
     :ready-to-first-token (summarize-ms (mapv :ready-to-first-token-ms samples))
     :decode (assoc decode-summary
                    :tokens-per-second (/ 1000.0 (:median-ms decode-summary)))
     :total (summarize-ms (mapv :total-ms samples))
     :prefix-load (when (every? :prefix-load-ms samples)
                    (summarize-ms (mapv :prefix-load-ms samples)))
     :prefix-load-phases (summarize-phase-timings samples)
     :prefix-transfer (when (seq transfer-samples)
                        {:samples transfer-samples
                         :totals (sum-transfer-differences transfer-samples)})}))

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

(defn cache-policy-calibration
  "Derive a worker-observation calibration from a continuation benchmark.

  The benchmark must be a result from `benchmark-paged-continuation!`. `tier`
  names the lower storage tier exercised by the benchmark and defaults to
  `:ssd`. The returned `:worker-observation-patch` can be merged into the
  observation consumed by the cluster candidate planner. Throughputs are
  one-point process-warm estimates, not claims about cold storage: the measured
  upload wall time is separated from the remaining prefix-load time and both
  affine intercepts are assumed to be zero.

  The result also reports checkpoint interference and a simple reuse break-even
  estimate. Missing transfer telemetry is tolerated; fields that cannot be
  derived are omitted from the observation patch."
  ([benchmark-result]
   (cache-policy-calibration benchmark-result :ssd))
  ([benchmark-result tier]
   (let [processed (long (get-in benchmark-result
                                 [:prompt :processed-token-count] 0))
         iterations (long (get-in benchmark-result
                                  [:restored :warm :iterations] 0))
         prefill-ms (get-in benchmark-result
                            [:uncached :warm :prompt-completion :median-ms])
         first-token-ms (get-in benchmark-result
                                [:uncached :warm :first-token :median-ms])
         prefix-load-ms (get-in benchmark-result
                                [:restored :warm :prefix-load :median-ms])
         transfer (get-in benchmark-result
                          [:restored :warm :prefix-transfer :totals])
         transferred-bytes (when transfer (transfer-total transfer :bytes))
         upload-wall-ms (when (and transfer (pos? iterations))
                          (/ (double (transfer-total transfer :host-wall-ns))
                             1.0e6
                             (double iterations)))
         bytes-per-sample (when (and transferred-bytes (pos? iterations))
                            (/ (double transferred-bytes) (double iterations)))
         lower-tier-ms (when (and prefix-load-ms upload-wall-ms)
                         (max 0.0 (- (double prefix-load-ms) upload-wall-ms)))
         uncached-ready (get-in benchmark-result
                                [:uncached :warm :ready-to-first-token :median-ms])
         restored-ready (get-in benchmark-result
                                [:restored :warm :ready-to-first-token :median-ms])
         saved-ms (when (and uncached-ready restored-ready)
                    (- (double uncached-ready) (double restored-ready)))
         capture-ms (get-in benchmark-result [:checkpoint :capture-total-ms])
         baseline-step (get-in benchmark-result
                               [:uncached :warm :decode :median-ms])
         overlap-step (get-in benchmark-result
                              [:checkpoint :inference-overlap
                               :step-latency :median-ms])
         interference-ms (when (and baseline-step overlap-step)
                           (max 0.0 (- (double overlap-step)
                                       (double baseline-step))))
         patch
         (cond-> {}
           (and (pos? processed) (number? prefill-ms))
           (assoc :worker/prefill-ms-per-token
                  (/ (double prefill-ms) (double processed)))

           (number? first-token-ms)
           (assoc :worker/first-token-ms (double first-token-ms))

           (and bytes-per-sample upload-wall-ms (pos? upload-wall-ms))
           (assoc :worker/gpu-restore-bytes-per-ms
                  (/ bytes-per-sample upload-wall-ms))

           (and bytes-per-sample lower-tier-ms (pos? lower-tier-ms))
           (assoc-in [:worker/tier-throughput-bytes-per-ms tier]
                     (/ bytes-per-sample lower-tier-ms)))]
     {:basis {:cache-state :process-and-page-cache-warm
              :cold-storage-measured? false
              :tier tier
              :processed-token-count processed
              :bytes-per-sample bytes-per-sample
              :prefix-load-ms prefix-load-ms
              :gpu-upload-wall-ms upload-wall-ms
              :lower-tier-and-control-ms lower-tier-ms}
      :worker-observation-patch patch
      :checkpoint-admission
      {:capture-total-ms capture-ms
       :checkpoint-ms capture-ms
       :foreground-interference-ms-per-step interference-ms
       :saved-ready-to-first-token-ms saved-ms
       :saved-ms-per-reuse saved-ms
       :break-even-reuses
       (when (and (number? capture-ms) saved-ms (pos? saved-ms))
         (/ (double capture-ms) saved-ms))}})))

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
     :phase-timings (some-> (:phase-timings ticket) deref)
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
             :cache-temperature {:first :first-process-use
                                 :warm :process-and-page-cache-warm
                                 :cold-storage-measured? false}
             :restore {:first-measured-ms first-ms
                       :first-process-use-ms first-ms
                       :warm warm-restore
                       :process-warm warm-restore}
             :speedup {:first-measured (/ prefill-median first-ms)
                       :warm (/ prefill-median warm-median)}
             :cache-stats (manager/stats cache)}
      probe (assoc :probe probe))))

(defn benchmark-paged-prefix!
  "Benchmark paged prompt computation against durable prefix restoration.

  The cache manager and paged decoder must already be open. Every sample uses a
  fresh continuation identity and releases its route afterwards. Restore timing
  includes mmap/scatter plus `prime-prompt!`, so a partial hit also measures
  computation of the exact uncached suffix. Checkpoint submission, local
  capture, and durable publication remain separately visible."
  [cache decoder model-fingerprint prompt-ids
   {:keys [iterations warmups probe-prompt-ids]
    :or {iterations 5 warmups 1}}]
  (let [pool (:pool decoder)
        prompt-ids (vec prompt-ids)
        with-route
        (fn [operation]
          (let [continuation-id (random-uuid)]
            (try
              (operation continuation-id)
              (finally
                (when (page-pool/route pool continuation-id)
                  (page-pool/release-route! pool continuation-id))))))
        prefill-operation
        #(with-route (fn [continuation-id]
                       (paged-decoder/prime-prompt! decoder continuation-id prompt-ids)))
        _ (prefill-operation)
        prefill (measure! prefill-operation
                          {:iterations iterations :warmups warmups})
        source-id (random-uuid)
        transfer-capabilities (page-pool/transfer-capabilities pool)
        _ (paged-decoder/prime-prompt! decoder source-id prompt-ids)
        checkpoint (checkpoint-paged! cache pool source-id model-fingerprint
                                      prompt-ids nil 0 transfer-capabilities)
        _ (page-pool/release-route! pool source-id)
        restore-ready
        (fn [tokens]
          (with-route
            (fn [continuation-id]
              (let [restored (manager/restore-paged-prefix!
                              cache pool continuation-id model-fingerprint tokens)]
                (paged-decoder/prime-prompt! decoder continuation-id tokens)
                restored))))
        restore #(restore-ready prompt-ids)
        first-restore (timed restore)
        warm-restore (measure! restore {:iterations iterations :warmups warmups})
        prefill-median (:median-ms prefill)
        first-ms (:milliseconds first-restore)
        warm-median (:median-ms warm-restore)
        probe
        (when probe-prompt-ids
          (let [probe-prompt-ids (vec probe-prompt-ids)
                result (timed #(restore-ready probe-prompt-ids))]
            {:milliseconds (:milliseconds result)
             :cached-token-count (get-in result [:value :cached-token-count])
             :processed-token-count (dec (count probe-prompt-ids))
             :speedup-vs-prefill (/ prefill-median (:milliseconds result))}))]
    (cond->
     {:prompt {:logical-token-count (count prompt-ids)
               :processed-token-count (dec (count prompt-ids))}
      :attention-execution (paged-decoder/attention-execution decoder)
      :transfer-capabilities transfer-capabilities
      :checkpoint checkpoint
      :prefill prefill
      :cache-temperature {:first :first-process-use
                          :warm :process-and-page-cache-warm
                          :cold-storage-measured? false}
      :restore {:first-measured-ms first-ms
                :first-process-use-ms first-ms
                :warm warm-restore
                :process-warm warm-restore}
      :speedup {:first-measured (/ prefill-median first-ms)
                :warm (/ prefill-median warm-median)}
      :cache-stats (manager/stats cache)}
      probe (assoc :probe probe))))

(defn benchmark-paged-continuation!
  "Benchmark a paged continuation through readiness and autoregressive decode.

  The cache manager and single-lane paged decoder must already be open and the
  model graph must already be bound. This function warms graph execution, stores
  one prompt prefix, and compares uncached prompt processing with process-warm
  restoration of the same prefix. Each sample reports prefix loading, uncached
  suffix completion, time to first generated token, and every subsequent decode
  step with its absolute position and visible context size.

  `opts` accepts positive `:iterations` (default 5), non-negative `:warmups`
  (default 1), positive `:decode-tokens` (default 4), and non-negative
  `:checkpoint-overlap-decode-tokens` (default 0). When overlap is requested, a
  second resident prompt decodes while the source route is captured; the result
  records how many steps started before capture completed and their latency. It
  classifies this as `:eligible` only when Raster reports device events on an
  independent physical transfer queue; otherwise it is an explicit
  `:interference-only` measurement. This requires capacity for both routes. The
  first restored sample is kept separate because it can include first-use
  storage and mmap costs. No
  operating-system caches are dropped, so it is not a cold-SSD result.

  Returns serializable raw samples, phase summaries, checkpoint drain costs,
  continuation speedups, and cache-manager counters. Every temporary resident
  route is released, including when execution fails."
  [cache decoder model-fingerprint prompt-ids
   {:keys [iterations warmups decode-tokens checkpoint-overlap-decode-tokens]
    :or {iterations 5 warmups 1 decode-tokens 4
         checkpoint-overlap-decode-tokens 0}}]
  (when-not (and (pos? iterations) (not (neg? warmups)) (pos? decode-tokens)
                 (not (neg? checkpoint-overlap-decode-tokens)))
    (throw (ex-info "Continuation benchmark counts are invalid"
                    {:iterations iterations
                     :warmups warmups
                     :decode-tokens decode-tokens
                     :checkpoint-overlap-decode-tokens
                     checkpoint-overlap-decode-tokens})))
  (when-not (= 1 (long (get-in decoder [:decode-state :batch-size] 1)))
    (throw (ex-info "Continuation benchmark currently requires a single decode lane"
                    {:batch-size (get-in decoder [:decode-state :batch-size])})))
  (let [pool (:pool decoder)
        transfer-capabilities (page-pool/transfer-capabilities pool)
        prompt-ids (vec prompt-ids)
        _ (when-not (seq prompt-ids)
            (throw (ex-info "Continuation benchmark requires a nonempty prompt" {})))
        preparation
        (timed #(page-pool/prepare-block-transfer!
                 pool
                 (min (long (or (:chunk-size cache)
                                (max 1 (dec (count prompt-ids)))))
                      (max 1 (dec (count prompt-ids))))))
        uncached-operation #(continuation-sample!
                             decoder prompt-ids (long decode-tokens) nil)
        _ (uncached-operation)
        uncached-first (uncached-operation)
        uncached-warm (continuation-series!
                       uncached-operation iterations warmups)
        source-id (random-uuid)
        overlap-id (when (pos? checkpoint-overlap-decode-tokens)
                     (random-uuid))
        checkpoint (try
                     (paged-decoder/prime-prompt! decoder source-id prompt-ids)
                     (when overlap-id
                       (paged-decoder/prime-prompt! decoder overlap-id prompt-ids))
                     (checkpoint-paged!
                      cache pool source-id model-fingerprint prompt-ids
                      (when overlap-id
                        (fn [index]
                          (paged-decoder/step!
                           decoder overlap-id
                           (+ (dec (count prompt-ids)) (long index)))))
                      checkpoint-overlap-decode-tokens
                      transfer-capabilities)
                     (finally
                       (when (page-pool/route pool source-id)
                         (page-pool/release-route! pool source-id))
                       (when (and overlap-id (page-pool/route pool overlap-id))
                         (page-pool/release-route! pool overlap-id))))
        restore! #(manager/restore-paged-prefix!
                   cache pool % model-fingerprint prompt-ids)
        restored-operation #(continuation-sample!
                              decoder prompt-ids (long decode-tokens) restore!)
        restored-first (restored-operation)
        restored-warm (continuation-series!
                       restored-operation iterations warmups)
        uncached-ready (get-in uncached-warm
                               [:ready-to-first-token :median-ms])
        restored-ready (get-in restored-warm
                               [:ready-to-first-token :median-ms])
        restored-first-ready (:ready-to-first-token-ms restored-first)]
    {:prompt {:logical-token-count (count prompt-ids)
              :processed-token-count (dec (count prompt-ids))}
     :decode-tokens (long decode-tokens)
     :transfer-capabilities transfer-capabilities
     :attention-execution (paged-decoder/attention-execution decoder)
     :block-transfer-preparation
     (assoc (:value preparation) :milliseconds (:milliseconds preparation))
     :checkpoint checkpoint
     :uncached {:first-measured uncached-first
                :warm uncached-warm}
     :cache-temperature {:first :first-process-use
                         :warm :process-and-page-cache-warm
                         :cold-storage-measured? false}
     :restored {:first-measured restored-first
                :first-process-use restored-first
                :warm restored-warm
                :process-warm restored-warm}
     :speedup {:first-restored-vs-uncached-warm
               (/ uncached-ready restored-first-ready)
               :warm-restored-vs-uncached-warm
               (/ uncached-ready restored-ready)}
     :cache-stats (manager/stats cache)}))

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
