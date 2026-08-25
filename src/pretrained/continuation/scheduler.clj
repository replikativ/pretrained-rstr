(ns pretrained.continuation.scheduler
  "Pure iteration and cache-source planning for paged decoder inference.

  The planner does not submit GPU work. It turns ready requests and measured
  costs into bounded work batches that a device-local runtime can execute with
  stable Raster graphs. Keeping this layer pure makes policy decisions replayable
  in a REPL and suitable for later persistence as Datahike observations.")

(def ^:private phases #{:decode :prefill :repair})

(defn request
  "Validate and normalize one schedulable request.

  Required keys are `:request/id`, `:request/phase`, and positive
  `:request/remaining-tokens`. `:request/arrival` defaults to zero. Decode work
  is always issued one token per iteration; prefill and repair work may be
  chunked by `plan-iteration`."
  [{:request/keys [id phase remaining-tokens arrival] :as value}]
  (when (nil? id)
    (throw (ex-info "Scheduled request requires :request/id" {:request value})))
  (when-not (contains? phases phase)
    (throw (ex-info "Scheduled request has an unsupported phase"
                    {:request value :supported phases})))
  (when-not (and (integer? remaining-tokens) (pos? remaining-tokens))
    (throw (ex-info "Scheduled request requires positive remaining tokens"
                    {:request value})))
  (when-not (or (nil? arrival) (and (integer? arrival) (not (neg? arrival))))
    (throw (ex-info "Scheduled request arrival must be a non-negative integer"
                    {:request value})))
  (assoc value :request/arrival (long (or arrival 0))))

(defn- phase-rank
  [phase]
  (case phase :decode 0 :repair 1 :prefill 2))

(defn- scheduled-tokens
  [request available chunk-size]
  (min (long available)
       (long (:request/remaining-tokens request))
       (if (= :decode (:request/phase request)) 1 (long chunk-size))))

(defn plan-iteration
  "Plan one continuous-batching iteration within fixed graph capacities.

  `opts` requires positive `:max-batched-tokens` and `:max-sequences` and may
  set positive `:prefill-chunk-size` and `:repair-chunk-size` (both default 128).
  Decode lanes are selected first to protect inter-token latency. A non-negative
  `:minimum-prefill-tokens` optionally reserves that token budget and one lane
  whenever repair/prefill work is waiting, preventing starvation under sustained
  decode load. The result contains `:scheduled`, `:deferred`, and unused
  capacities; each scheduled item has `:scheduled/tokens` but input is not mutated."
  [{:keys [max-batched-tokens max-sequences prefill-chunk-size repair-chunk-size
           minimum-prefill-tokens]
    :or {prefill-chunk-size 128 repair-chunk-size 128 minimum-prefill-tokens 0}}
   requests]
  (doseq [[field value] [[:max-batched-tokens max-batched-tokens]
                         [:max-sequences max-sequences]
                         [:prefill-chunk-size prefill-chunk-size]
                         [:repair-chunk-size repair-chunk-size]]]
    (when-not (and (integer? value) (pos? value))
      (throw (ex-info "Scheduler capacities must be positive integers"
                      {:field field :value value}))))
  (when-not (and (integer? minimum-prefill-tokens)
                 (not (neg? minimum-prefill-tokens)))
    (throw (ex-info "Minimum prefill reservation must be a non-negative integer"
                    {:minimum-prefill-tokens minimum-prefill-tokens})))
  (let [ordered (sort-by (juxt (comp phase-rank :request/phase)
                               :request/arrival
                               (comp str :request/id))
                         (mapv request requests))
        nondecode? (some #(not= :decode (:request/phase %)) ordered)
        reserved-tokens (if nondecode?
                          (min (long minimum-prefill-tokens)
                               (long max-batched-tokens))
                          0)
        reserved-lane? (and nondecode? (pos? reserved-tokens))]
    (loop [pending ordered
           token-budget (long max-batched-tokens)
           sequence-budget (long max-sequences)
           scheduled []
           deferred []]
      (if-let [item (first pending)]
        (let [decode? (= :decode (:request/phase item))
              available-tokens (if decode?
                                 (max 0 (- token-budget reserved-tokens))
                                 token-budget)
              available-sequences (if (and decode? reserved-lane?)
                                    (max 0 (dec sequence-budget))
                                    sequence-budget)]
          (if (or (zero? available-tokens) (zero? available-sequences))
            (recur (next pending) token-budget sequence-budget
                   scheduled (conj deferred item))
            (let [chunk-size (if (= :repair (:request/phase item))
                               repair-chunk-size
                               prefill-chunk-size)
                  tokens (scheduled-tokens item available-tokens chunk-size)]
              (recur (next pending)
                     (- token-budget tokens)
                     (dec sequence-budget)
                     (conj scheduled (assoc item :scheduled/tokens tokens))
                     deferred))))
        {:scheduled scheduled
         :deferred deferred
         :unused-token-capacity token-budget
         :unused-sequence-capacity sequence-budget}))))

(defn advance-request
  "Apply one scheduled item and return its next request, or nil when complete."
  [{:request/keys [remaining-tokens] :scheduled/keys [tokens] :as scheduled}]
  (when-not (and (integer? tokens) (pos? tokens) (<= tokens remaining-tokens))
    (throw (ex-info "Scheduled token count is outside the request"
                    {:scheduled scheduled})))
  (let [remaining (- remaining-tokens tokens)]
    (when (pos? remaining)
      (-> scheduled
          (dissoc :scheduled/tokens)
          (assoc :request/remaining-tokens remaining)))))

(defn choose-cache-source
  "Choose the lowest-latency eligible way to establish a prompt continuation.

  Candidates require `:source/kind` and non-negative `:source/estimated-ms`.
  Exact candidates are eligible by default. Approximate candidates additionally
  require `:source/quality` at least `:minimum-quality` and are disabled unless
  `:allow-approximate?` is true. Deterministic kind rank breaks equal-cost ties."
  ([candidates] (choose-cache-source candidates {}))
  ([candidates {:keys [allow-approximate? minimum-quality]
                :or {allow-approximate? false minimum-quality 1.0}}]
   (when-not (and (number? minimum-quality)
                  (<= 0.0 (double minimum-quality) 1.0))
     (throw (ex-info "Minimum cache-source quality must be in [0,1]"
                     {:minimum-quality minimum-quality})))
   (let [rank {:resident-exact 0 :restore-exact 1 :recompute-exact 2
               :modular-repair 3}
         checked
         (mapv (fn [{:source/keys [kind estimated-ms exact? quality] :as candidate}]
                 (when-not (contains? rank kind)
                   (throw (ex-info "Unknown cache source kind" {:candidate candidate})))
                 (when-not (and (number? estimated-ms)
                                (not (neg? (double estimated-ms))))
                   (throw (ex-info "Cache source requires a non-negative estimate"
                                   {:candidate candidate})))
                 (assoc candidate
                        :source/exact? (if (nil? exact?)
                                         (not= kind :modular-repair)
                                         exact?)
                        :source/quality (double (or quality
                                                   (if (= kind :modular-repair)
                                                     0.0 1.0)))))
               candidates)
         eligible (filter (fn [{:source/keys [exact? quality]}]
                            (or exact?
                                (and allow-approximate?
                                     (>= quality (double minimum-quality)))))
                          checked)]
     (first (sort-by (juxt :source/estimated-ms
                           (comp rank :source/kind))
                     eligible)))))
