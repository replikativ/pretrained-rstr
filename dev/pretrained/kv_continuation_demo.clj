(ns pretrained.kv-continuation-demo
  "REPL showcases for durable CPU round-trips and asynchronous GPU checkpoints."
  (:require [pretrained.continuation :as continuation]
            [pretrained.continuation.benchmark :as benchmark]
            [pretrained.continuation.gpu :as continuation-gpu]
            [pretrained.continuation.manager :as manager]
            [pretrained.continuation.page-pool :as page-pool]
            [pretrained.continuation.paged-decoder :as paged-decoder]
            [pretrained.continuation.residency :as residency]
            [pretrained.continuation.scheduler :as scheduler]
            [pretrained.decoder-gpu :as decoder-gpu]
            [pretrained.model-identity :as model-identity]
            [raster.gpu.core :as gpu]))

(defn- fingerprint
  [model opts]
  (or (:model-fingerprint opts)
      (model-identity/compatibility-fingerprint
       model (select-keys opts [:weights-id :execution-variant]))))

(defn- timed
  [f]
  (let [started (System/nanoTime)
        value (f)]
    {:value value :milliseconds (/ (- (System/nanoTime) started) 1.0e6)}))

(defn plan-serving-iteration
  "Plan one illustrative Gemma serving iteration and exact cache-source choice.

  `requests` follow `pretrained.continuation.scheduler/request`. `candidates`
  contain measured resident, restore, and recompute costs. This helper performs
  no GPU work; it is intended for changing budgets and measurements at a REPL
  before connecting the same decisions to the resident decoder executor."
  [requests candidates opts]
  {:batch (scheduler/plan-iteration opts requests)
   :cache-source (scheduler/choose-cache-source candidates)})

(defn plan-page-admission
  "Explain whether a page-pool snapshot can admit a Gemma continuation.

  Only durable, unpinned, unleased routes are considered for eviction. The
  function is pure and does not alter the supplied snapshot."
  [snapshot token-count opts]
  (residency/plan-admission snapshot token-count opts))

(defn run-cpu-roundtrip!
  "Run an uninterrupted/split continuation comparison and catalog the checkpoint.

  Intended for an nREPL with `:dev`. `opts` requires `:max-position`,
  `:split-tokens`, and `:tail-tokens`. It derives a checkpoint fingerprint unless
  `:model-fingerprint` is supplied. Returns token-exactness,
  the Datahike entity and separate checkpoint/query/restore timings."
  [model prompt-ids datahike-config cache-directory opts]
  (let [{:keys [max-position split-tokens tail-tokens]} opts
        model-fingerprint (fingerprint model opts)
        cache (manager/open-manager datahike-config cache-directory)]
    (try
      (let [total (+ (long split-tokens) (long tail-tokens))
            uninterrupted (continuation/advance-cpu
                           (continuation/start-cpu model prompt-ids
                                                   {:max-position max-position
                                                    :model-fingerprint model-fingerprint})
                           total)
            prefix (continuation/advance-cpu
                    (continuation/start-cpu model prompt-ids
                                            {:max-position max-position
                                             :model-fingerprint model-fingerprint})
                    split-tokens)
            checkpoint (timed #(manager/checkpoint-cpu! cache (:continuation prefix)))
            logical-prefix (:continuation/tokens (:continuation prefix))
            query (timed #(manager/lookup cache model-fingerprint logical-prefix))
            restore (timed #(manager/restore-cpu (:value query) model
                                                 {:max-position max-position
                                                  :model-fingerprint model-fingerprint}))
            suffix (continuation/advance-cpu (:value restore) tail-tokens)
            split-output (into (:tokens prefix) (:tokens suffix))]
        {:token-exact? (= (:tokens uninterrupted) split-output)
         :uninterrupted-tokens (:tokens uninterrupted)
         :split-tokens split-output
         :entry (:value checkpoint)
         :checkpoint-ms (:milliseconds checkpoint)
         :query-ms (:milliseconds query)
         :restore-ms (:milliseconds restore)
         :cache-stats (manager/stats cache)})
      (finally
        (.close cache)))))

(defn run-gpu-async-checkpoints!
  "Generate on a resident decoder while checkpoint capture/publication run behind it.

  `opts` requires `:tokens` and `:checkpoint-every`; it derives a checkpoint
  fingerprint unless one is supplied. It may
  contain `:checkpoint-final?`, `:chunked?`, and `:chunk-size`. The reported
  generation time excludes waiting for cache work. The function subsequently
  waits for accepted tickets so its return value reports durable publication and
  it can safely close the manager.

  Level Zero capture is a host copy from coherent shared memory. Raster's current
  OpenCL path performs blocking reads on its device queue, so an NVIDIA OpenCL run
  is caller-asynchronous but may still contend with inference until a native CUDA
  copy-stream/event path is available."
  [dstate prompt-ids datahike-config cache-directory opts]
  (let [{:keys [tokens checkpoint-every checkpoint-final?
                chunked? chunk-size]} opts
        model-fingerprint (fingerprint (:model dstate) opts)
        cache (manager/open-manager datahike-config cache-directory
                                    (cond-> {} chunk-size (assoc :chunk-size chunk-size)))]
    (try
      (let [initial (continuation-gpu/start-gpu
                     dstate prompt-ids {:model-fingerprint model-fingerprint})
            generation (timed #(manager/advance-gpu-with-checkpoints!
                                cache initial tokens
                                {:checkpoint-every checkpoint-every
                                 :checkpoint-final? checkpoint-final?
                                 :chunked? chunked?}))
            tickets (:checkpoints (:value generation))
            accepted (filter :accepted? tickets)
            captured (timed #(mapv (fn [ticket] (.get (:captured ticket))) accepted))
            published (timed #(mapv (fn [ticket] (.get (:published ticket))) accepted))]
        {:tokens (:tokens (:value generation))
         :generation-ms (:milliseconds generation)
         :checkpoint-count (count tickets)
         :accepted-count (count accepted)
         :capture-drain-ms (:milliseconds captured)
         :publication-drain-ms (:milliseconds published)
         :entries (:value published)
         :cache-stats (manager/stats cache)})
      (finally
        (.close cache)))))

(defn run-gpu-prefix-reuse!
  "Checkpoint a prompt as chunks, restore it, and compare generated tokens.

  `source-dstate` and `destination-dstate` should be independently bound decoder
  states for the same model. `opts` requires `:tail-tokens`; `:chunk-size` defaults
  to 256 and the model fingerprint is derived unless supplied. The returned timings separate the
  caller-asynchronous checkpoint submission/drain from query+mmap+GPU restoration."
  [source-dstate destination-dstate prompt-ids datahike-config cache-directory opts]
  (let [{:keys [tail-tokens chunk-size]} opts
        model-fingerprint (fingerprint (:model source-dstate) opts)
        cache (manager/open-manager datahike-config cache-directory
                                    (cond-> {} chunk-size (assoc :chunk-size chunk-size)))]
    (try
      (let [source (continuation-gpu/start-gpu
                    source-dstate prompt-ids {:model-fingerprint model-fingerprint})
            submit (timed #(manager/checkpoint-gpu-chunks-async! cache source))
            ticket (:value submit)
            capture (timed #(.get (:captured ticket)))
            publication (timed #(.get (:published ticket)))
            restore (timed #(manager/restore-gpu-prefix
                             cache destination-dstate model-fingerprint prompt-ids))
            uninterrupted (continuation-gpu/advance-gpu source tail-tokens)
            resumed (continuation-gpu/advance-gpu
                     (:continuation (:value restore)) tail-tokens)]
        {:token-exact? (= (:tokens uninterrupted) (:tokens resumed))
         :uninterrupted-tokens (:tokens uninterrupted)
         :resumed-tokens (:tokens resumed)
         :cached-token-count (:cached-token-count (:value restore))
         :matched-chunks (count (:matched (:value restore)))
         :submit-ms (:milliseconds submit)
         :capture-drain-ms (:milliseconds capture)
         :publication-drain-ms (:milliseconds publication)
         :restore-ms (:milliseconds restore)
         :cache-stats (manager/stats cache)})
      (finally
        (.close cache)))))

(defn run-paged-continuation-benchmark!
  "Benchmark a loaded decoder model through durable paged continuation reuse.

  Intended for an nREPL with `:dev`. Model loading is deliberately outside the
  measurement. `opts` requires `:max-position` and may contain `:page-size`,
  `:physical-pages`, `:chunk-size`, `:iterations`, `:warmups`,
  `:decode-tokens`, `:checkpoint-overlap-decode-tokens`,
  `:attention-schedule`, `:history-tile-size`, and the fingerprint options
  accepted by `fingerprint`. Overlap measurement needs GPU pages for two prompt
  routes plus generated tokens. `:device-id` defaults to Raster's `:ze:0` and
  may select an OpenCL device such as `:ocl:0`.

  Returns the instrumented benchmark plus the first restored token sequence.
  The function closes the temporary manager, paged decoder, and Raster session."
  [model prompt-ids datahike-config cache-directory opts]
  (let [{:keys [max-position page-size physical-pages chunk-size
                iterations warmups decode-tokens attention-schedule
                checkpoint-overlap-decode-tokens history-tile-size device-id]
         :or {page-size 16 iterations 5 warmups 1 decode-tokens 4
              device-id :ze:0}} opts
        model-fingerprint (fingerprint model opts)
        _ (when-not (and (integer? max-position) (pos? max-position))
            (throw (ex-info "Paged benchmark requires a positive :max-position"
                            {:max-position max-position})))
        cache (manager/open-manager
               datahike-config cache-directory
               (cond-> {} chunk-size (assoc :chunk-size chunk-size)))
        dstate (volatile! nil)
        decoder (volatile! nil)]
    (try
      (vreset! dstate
               (decoder-gpu/bind-decode!
                model :maxpos max-position :cache-mode :paged :batch-size 1
                :device-id device-id))
      (vreset! decoder
               (paged-decoder/open!
                @dstate
                :page-size page-size
                :physical-pages physical-pages
                :attention-schedule attention-schedule
                :history-tile-size history-tile-size))
      (let [result
            (benchmark/benchmark-paged-continuation!
             cache @decoder model-fingerprint (vec prompt-ids)
             {:iterations iterations
              :warmups warmups
              :decode-tokens decode-tokens
              :checkpoint-overlap-decode-tokens
              (or checkpoint-overlap-decode-tokens 0)})]
        (assoc result
               :first-restored-tokens
               (mapv :token (get-in result [:restored :first-measured :steps]))))
      (finally
        (try
          (when @decoder
            (try
              (paged-decoder/close! @decoder)
              (finally
                (page-pool/close-transfer-engines! (:pool @decoder)))))
          (finally
            (when @dstate (gpu/close-session! (:sess @dstate)))
            (.close cache)))))))
