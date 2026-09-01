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

(defn plan-decode-lanes
  "Retain and refill a worker's fixed-capacity decode lanes.

  `previous-lanes` is a vector of request maps or nil, indexed by physical
  execution lane. `requests` contains the current runnable queue. Decode
  requests already occupying a lane retain it; vacancies are filled by arrival
  and stable request identity. The result contains the complete `:lanes`, newly
  assigned `:refill`, removed `:retired`, unassigned `:deferred`, and the active
  continuation identities that GPU residency must protect from eviction."
  [capacity previous-lanes requests]
  (when-not (and (integer? capacity) (pos? capacity))
    (throw (ex-info "Decode lane capacity must be a positive integer"
                    {:capacity capacity})))
  (let [capacity (long capacity)
        previous-lanes (vec previous-lanes)
        _ (when (> (count previous-lanes) capacity)
            (throw (ex-info "Previous lane table exceeds decoder capacity"
                            {:capacity capacity :lane-count (count previous-lanes)})))
        previous-lanes (into previous-lanes
                             (repeat (- capacity (count previous-lanes)) nil))
        runnable (->> requests
                      (mapv request)
                      (filterv #(= :decode (:request/phase %))))
        by-id (into {} (map (juxt :request/id identity)) runnable)
        retained
        (mapv (fn [lane prior]
                (when-let [current (and prior (get by-id (:request/id prior)))]
                  (assoc current :lane/index lane)))
              (range capacity) previous-lanes)
        retained-ids (into #{} (keep :request/id) retained)
        waiting (->> runnable
                     (remove #(contains? retained-ids (:request/id %)))
                     (sort-by (juxt :request/arrival (comp str :request/id))))
        [lanes remaining]
        (reduce (fn [[lanes waiting] lane]
                  (if (or (get lanes lane) (empty? waiting))
                    [lanes waiting]
                    [(assoc lanes lane (assoc (first waiting) :lane/index lane))
                     (next waiting)]))
                [retained waiting]
                (range capacity))
        refill (filterv #(and % (not (contains? retained-ids (:request/id %)))) lanes)
        lane-ids (into #{} (keep :request/id) lanes)
        retired (filterv #(and % (not (contains? lane-ids (:request/id %))))
                         previous-lanes)
        protected (into #{}
                        (keep #(or (:request/continuation-id %)
                                   (:request/id %)))
                        lanes)]
    {:lanes lanes
     :retained (filterv some? retained)
     :refill refill
     :retired retired
     :deferred (vec remaining)
     :protected-continuation-ids protected}))

(defn plan-work-lanes
  "Retain and refill fixed execution lanes for an already ordered work set.

  Unlike `plan-decode-lanes`, this planner accepts decode, prefill, and repair
  items and preserves the order chosen by `plan-iteration`. Existing selected
  requests retain their physical lane; vacancies receive scheduled items in
  input order. The result reports `:lanes`, `:retained`, `:refill`, `:retired`,
  `:deferred`, and continuation identities that residency must protect."
  [capacity previous-lanes scheduled]
  (when-not (and (integer? capacity) (pos? capacity))
    (throw (ex-info "Work lane capacity must be a positive integer"
                    {:capacity capacity})))
  (let [capacity (long capacity)
        previous-lanes (vec previous-lanes)
        _ (when (> (count previous-lanes) capacity)
            (throw (ex-info "Previous lane table exceeds execution capacity"
                            {:capacity capacity
                             :lane-count (count previous-lanes)})))
        previous-lanes (into previous-lanes
                             (repeat (- capacity (count previous-lanes)) nil))
        scheduled (mapv request scheduled)
        ids (mapv :request/id scheduled)
        _ (when-not (= (count ids) (count (set ids)))
            (throw (ex-info "Scheduled work identities must be unique"
                            {:request-ids ids})))
        by-id (into {} (map (juxt :request/id identity)) scheduled)
        retained
        (mapv (fn [lane prior]
                (when-let [current (and prior (get by-id (:request/id prior)))]
                  (assoc current :lane/index lane)))
              (range capacity) previous-lanes)
        retained-ids (into #{} (keep :request/id) retained)
        waiting (remove #(contains? retained-ids (:request/id %)) scheduled)
        [lanes remaining]
        (reduce (fn [[lanes waiting] lane]
                  (if (or (get lanes lane) (empty? waiting))
                    [lanes waiting]
                    [(assoc lanes lane (assoc (first waiting) :lane/index lane))
                     (next waiting)]))
                [retained waiting]
                (range capacity))
        refill (filterv #(and % (not (contains? retained-ids (:request/id %))))
                        lanes)
        lane-ids (into #{} (keep :request/id) lanes)]
    {:lanes lanes
     :retained (filterv some? retained)
     :refill refill
     :retired (filterv #(and % (not (contains? lane-ids (:request/id %))))
                       previous-lanes)
     :deferred (vec remaining)
     :protected-continuation-ids
     (into #{}
           (keep #(or (:request/continuation-id %) (:request/id %)))
           lanes)}))

(defn decode-submission
  "Translate a lane plan into paged-decoder work and selective priming rows.

  Active requests require `:request/continuation-id` and non-negative
  `:request/position`; newly filled lanes additionally require
  `:request/pending-token`. The returned maps can be passed directly to
  `paged-decoder/prime-lanes!` and `paged-decoder/step-lanes!`."
  [{:keys [lanes refill] :as plan}]
  (let [active (filterv some? lanes)
        refill-ids (into #{} (map :request/id) refill)]
    (doseq [{:request/keys [id continuation-id position pending-token] :as item} active]
      (when (nil? continuation-id)
        (throw (ex-info "Decode lane requires a continuation identity"
                        {:request item})))
      (when-not (and (integer? position) (not (neg? position)))
        (throw (ex-info "Decode lane requires a non-negative position"
                        {:request item})))
      (when (and (contains? refill-ids id) (nil? pending-token))
        (throw (ex-info "A refilled decode lane requires a pending token"
                        {:request item}))))
    {:lane-work
     (mapv (fn [{:keys [lane/index]
                 :request/keys [continuation-id position]}]
             {:lane index :continuation-id continuation-id :position position})
           active)
     :prime-lanes
     (into []
           (comp (filter #(contains? refill-ids (:request/id %)))
                 (map (fn [{:keys [lane/index]
                            :request/keys [pending-token]}]
                        {:lane index :token pending-token})))
           active)
     :protected-continuation-ids (:protected-continuation-ids plan)}))

(defn complete-decode-iteration
  "Apply lane-ordered decoder results and identify retained/completed requests.

  Every active lane must have one `{:lane index :token token-id}` result.
  Remaining-token budgets decrease by one, positions advance by one, and the
  emitted token becomes the next pending token. A request completes on exhausted
  budget or an id in `:eos-ids`. The returned lane table can be supplied as
  `previous-lanes` to the next `plan-decode-lanes` call."
  ([plan results] (complete-decode-iteration plan results {}))
  ([{:keys [lanes]} results {:keys [eos-ids] :or {eos-ids #{}}}]
   (let [results (vec results)
         by-lane (into {} (map (juxt :lane identity)) results)
         active-lanes (into #{} (keep :lane/index) lanes)]
     (when-not (and (= (count results) (count by-lane))
                    (= active-lanes (set (keys by-lane))))
       (throw (ex-info "Decode results do not cover active lanes exactly"
                       {:active-lanes active-lanes
                        :result-lanes (mapv :lane results)})))
     (let [updated
           (mapv (fn [item]
                   (when item
                     (let [token (:token (get by-lane (:lane/index item)))]
                       (when-not (integer? token)
                         (throw (ex-info "Decode result requires an integer token"
                                         {:lane (:lane/index item) :token token})))
                       (-> item
                           (update :request/remaining-tokens dec)
                           (update :request/position inc)
                           (assoc :request/pending-token token
                                  :iteration/token token)))))
                 lanes)
           complete? #(and %
                           (or (zero? (:request/remaining-tokens %))
                               (contains? eos-ids (:iteration/token %))))
           completed (filterv complete? updated)
           next-lanes (mapv #(when-not (complete? %) (dissoc % :iteration/token))
                            updated)]
       {:lanes next-lanes
        :runnable (filterv some? next-lanes)
        :completed completed}))))

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
