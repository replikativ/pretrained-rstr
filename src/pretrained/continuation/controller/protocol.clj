(ns pretrained.continuation.controller.protocol
  "Validated values shared by continuation routers, workers, and interpreters.

  The protocol is transport-neutral. Kabel messages, an HTTP ingress, and the
  deterministic simulator all carry these values without gaining authority to
  mutate GPU or storage state directly."
  (:require [clojure.string :as str]))

(def cache-tiers
  "Exact-prefix locations understood by the initial routing policy."
  #{:gpu :ram :ssd :peer :object :none})

(defn generation-request
  "Validate and normalize a consumer generation request.

  Required keys are `:request/id`, `:request/model-fingerprint`, a nonempty
  integer `:request/tokens` collection, and positive
  `:request/max-new-tokens`. `:request/arrival` and `:request/priority` default
  to zero. The returned request is immutable protocol data suitable for Kabel
  or an HTTP adapter.

  Throws `ExceptionInfo` when a required field or bound is invalid."
  [{:request/keys [id model-fingerprint tokens max-new-tokens arrival priority]
    :as request}]
  (when (nil? id)
    (throw (ex-info "Generation request requires :request/id"
                    {:request request})))
  (when-not (and (string? model-fingerprint)
                 (not (str/blank? model-fingerprint)))
    (throw (ex-info "Generation request requires a model fingerprint"
                    {:request request})))
  (let [tokens (vec tokens)]
    (when-not (and (seq tokens) (every? integer? tokens))
      (throw (ex-info "Generation request requires nonempty integer tokens"
                      {:request request})))
    (when-not (and (integer? max-new-tokens) (pos? max-new-tokens))
      (throw (ex-info "Generation request requires positive max-new-tokens"
                      {:request request})))
    (doseq [[field value] [[:request/arrival (or arrival 0)]
                           [:request/priority (or priority 0)]]]
      (when-not (and (integer? value) (not (neg? value)))
        (throw (ex-info "Generation request scheduling fields must be non-negative integers"
                        {:field field :value value :request request}))))
    (assoc request
           :request/tokens tokens
           :request/max-new-tokens (long max-new-tokens)
           :request/arrival (long (or arrival 0))
           :request/priority (long (or priority 0)))))

(defn worker-candidate
  "Validate one request-specific worker routing candidate.

  Candidates are derived from a worker/load snapshot and an exact-prefix
  lookup. Required fields are `:candidate/worker-id`,
  `:candidate/worker-epoch`, `:candidate/cache-tier`, and non-negative cost and
  capacity fields. `:candidate/cached-token-count` excludes the pending final
  prompt token. `:candidate/exact?` defaults to true; approximate KV reuse is
  rejected by this protocol. Optional `:candidate/transfer-capabilities` carries
  the worker's physical transfer contract into request scheduling.

  Returns a normalized candidate or throws `ExceptionInfo`."
  [{:candidate/keys [worker-id worker-epoch cache-tier cached-token-count
                     cached-bytes queue-ms prefix-load-ms gpu-restore-ms
                     prefill-ms-per-token first-token-ms page-size free-pages
                     evictable-pages max-context online? model-loaded? exact?]
    :as candidate}]
  (when (nil? worker-id)
    (throw (ex-info "Worker candidate requires an identity"
                    {:candidate candidate})))
  (when-not (and (integer? worker-epoch) (not (neg? worker-epoch)))
    (throw (ex-info "Worker candidate requires a non-negative epoch"
                    {:candidate candidate})))
  (when-not (contains? cache-tiers cache-tier)
    (throw (ex-info "Worker candidate has an unsupported cache tier"
                    {:candidate candidate :supported cache-tiers})))
  (when (false? exact?)
    (throw (ex-info "Approximate KV candidates are not executable"
                    {:candidate candidate})))
  (when-not (or (nil? (:candidate/transfer-capabilities candidate))
                (map? (:candidate/transfer-capabilities candidate)))
    (throw (ex-info "Worker candidate transfer capabilities must be a map"
                    {:candidate candidate})))
  (doseq [[field value]
          [[:candidate/cached-token-count cached-token-count]
           [:candidate/cached-bytes cached-bytes]
           [:candidate/queue-ms queue-ms]
           [:candidate/prefix-load-ms prefix-load-ms]
           [:candidate/gpu-restore-ms gpu-restore-ms]
           [:candidate/prefill-ms-per-token prefill-ms-per-token]
           [:candidate/first-token-ms first-token-ms]
           [:candidate/page-size page-size]
           [:candidate/free-pages free-pages]
           [:candidate/evictable-pages evictable-pages]
           [:candidate/max-context max-context]]]
    (when-not (and (number? value) (not (neg? (double value))))
      (throw (ex-info "Worker candidate fields must be non-negative numbers"
                      {:field field :value value :candidate candidate}))))
  (when-not (and (integer? page-size) (pos? page-size))
    (throw (ex-info "Worker candidate page size must be positive"
                    {:candidate candidate})))
  (assoc candidate
         :candidate/cached-token-count (long cached-token-count)
         :candidate/cached-bytes (long cached-bytes)
         :candidate/worker-epoch (long worker-epoch)
         :candidate/page-size (long page-size)
         :candidate/free-pages (long free-pages)
         :candidate/evictable-pages (long evictable-pages)
         :candidate/max-context (long max-context)
         :candidate/online? (not (false? online?))
         :candidate/model-loaded? (not (false? model-loaded?))
         :candidate/transfer-capabilities
         (or (:candidate/transfer-capabilities candidate) {})
         :candidate/exact? true))

(defn assignment-id
  "Return a stable assignment-attempt identity for `request-id` and `attempt`."
  [request-id attempt]
  [request-id (long attempt)])
