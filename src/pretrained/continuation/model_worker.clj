(ns pretrained.continuation.model-worker
  "Assemble one real-model paged worker: decode session, page pool, continuation
  manager, and controller handlers, with ordered teardown.

  The worker is transport-neutral. Attach it to an in-process router with
  `pretrained.openai.local`, or to a Kabel router by passing `worker-opts`,
  `handlers`, and `measurements` to
  `pretrained.continuation.controller.kabel/open-worker-endpoint`. Close the
  controller or endpoint before closing the worker."
  (:require [pretrained.continuation.controller.paged :as paged-controller]
            [pretrained.continuation.manager :as manager]
            [pretrained.continuation.page-pool :as page-pool]
            [pretrained.continuation.paged-decoder :as paged-decoder]
            [pretrained.decoder-gpu :as decoder-gpu]
            [raster.gpu.core :as gpu])
  (:import (java.io Closeable)))

(defrecord ModelWorker
    [id decode-state decoder cache pool worker-opts handlers measurements
     closed?]
  Closeable
  (close [_]
    (when (compare-and-set! closed? false true)
      (try
        (when cache (.close ^Closeable cache))
        (finally
          (try
            (when decoder
              (try
                (paged-decoder/close! decoder)
                (finally
                  (page-pool/close-transfer-engines! (:pool decoder)))))
            (finally
              (when decode-state
                (gpu/close-session! (:sess decode-state))))))))))

(defn default-measurements
  "Return deployment-constant scheduling measurements for `worker-id`.

  These are conservative placeholders. A long-running worker should replace
  them with `pretrained.continuation.calibration/measurements-fn`."
  [worker-id max-position]
  {:worker/node (name worker-id)
   :worker/queue-ms 0.0
   :worker/max-context (long max-position)
   :worker/prefill-ms-per-token 10.0
   :worker/first-token-ms 10.0
   :worker/gpu-restore-bytes-per-ms 1000000.0
   :worker/tier-throughput-bytes-per-ms {}
   :worker/object-store? false})

(defn open-worker!
  "Open a paged worker for `model` with shared `quantized-weights`.

  Required options are `:fingerprint`, the model compatibility fingerprint, and
  either `:connection` (a Datahike catalog connection) or `:datahike-config`,
  plus `:cache-directory` for worker-local chunk files. Optional:

  - `:worker-id` (default `:worker-a`) and `:device-id` (default `:ze:0`);
  - `:max-position` (default 1024), `:page-size` (default 16), and
    `:physical-pages` (default four sequences of `:max-position`);
  - `:chunk-size` (default 16 processed tokens per durable chunk; resident
    prefixes are advertised at their exact end and at every chunk boundary);
  - `:eos-ids`, the generation stop tokens (default from the model);
  - `:checkpoint-policy`, passed to the paged handlers;
  - `:measurements`, a map or zero-argument function (default
    `default-measurements`); and
  - `:max-pending-captures` / `:max-pending-publications` (default 1); and
  - `:numerical-contract`, a structured execution variant's contract.

  Returns a `ModelWorker` whose `:pool`, `:worker-opts`, `:handlers`, and
  `:measurements` are ready for a controller."
  [model quantized-weights
   {:keys [fingerprint connection datahike-config cache-directory worker-id
           device-id max-position page-size physical-pages chunk-size eos-ids
           checkpoint-policy measurements max-pending-captures
           max-pending-publications numerical-contract]
    :or {worker-id :worker-a
         device-id :ze:0
         max-position 1024
         page-size 16
         chunk-size 16
         max-pending-captures 1
         max-pending-publications 1}}]
  (when-not (string? fingerprint)
    (throw (ex-info "Model worker requires a compatibility fingerprint" {})))
  (when-not (or connection datahike-config)
    (throw (ex-info "Model worker requires :connection or :datahike-config" {})))
  (when-not cache-directory
    (throw (ex-info "Model worker requires :cache-directory" {})))
  (let [max-position (long max-position)
        page-size (long page-size)
        pages-per-sequence (quot (+ max-position (dec page-size)) page-size)
        physical-pages (long (or physical-pages (* 4 pages-per-sequence)))
        eos-ids (set (or eos-ids (:eos-ids model) #{}))
        decode-state (volatile! nil)
        decoder (volatile! nil)
        cache (volatile! nil)]
    (try
      (vreset! decode-state
               (decoder-gpu/bind-decode!
                model :qw quantized-weights :maxpos max-position
                :cache-mode :paged :batch-size 1 :device-id device-id))
      (vreset! decoder
               (paged-decoder/open!
                @decode-state :page-size page-size
                :physical-pages physical-pages :key-prefix (name worker-id)))
      (vreset! cache
               (manager/open-manager
                datahike-config cache-directory
                (cond-> {:chunk-size (long chunk-size)
                         :max-pending-captures max-pending-captures
                         :max-pending-publications max-pending-publications}
                  connection (assoc :connection connection)
                  numerical-contract (assoc :numerical-contract numerical-contract))))
      (->ModelWorker
       worker-id @decode-state @decoder @cache (:pool @decoder)
       {:worker/id worker-id :worker/epoch 0
        :worker/models #{fingerprint}
        :worker/free-pages physical-pages :worker/evictable-pages 0}
       (paged-controller/handlers
        @cache @decoder
        (cond-> {:chunk-size (long chunk-size) :eos-ids eos-ids}
          checkpoint-policy (assoc :checkpoint-policy checkpoint-policy)))
       (or measurements (default-measurements worker-id max-position))
       (atom false))
      (catch Throwable error
        (try
          (.close (->ModelWorker worker-id @decode-state @decoder @cache nil
                                 nil nil nil (atom false)))
          (catch Throwable _))
        (throw error)))))
