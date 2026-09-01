(ns pretrained.continuation.controller.router
  "Pure locality- and load-aware routing for continuation requests.

  `transition` owns assignment attempts but performs no I/O. Every external
  action is returned as effect data so the same machine runs under Kabel, a
  deterministic simulator, or another transport such as Netz."
  (:require [pretrained.continuation.controller.protocol :as protocol]))

(def terminal-phases #{:completed :failed :cancelled})

(defn initial-state
  "Create an empty cluster-router state."
  []
  {:router/requests {}})

(defn- pages-for
  [token-count page-size]
  (if (zero? token-count)
    0
    (inc (quot (dec token-count) page-size))))

(defn estimate-candidate
  "Estimate ready-to-first-token cost for one request/candidate pair.

  Queueing and lower-tier prefix loading may overlap. GPU restoration follows
  the load, then the exact uncached suffix is prefetched and one decode step is
  executed. A candidate is infeasible when it is offline, lacks the model,
  exceeds context capacity, reports a prefix longer than the request, or cannot
  provide enough free plus evictable pages.

  The result also exposes `:estimate/live-transfer-overlap-eligible?` as an
  execution-policy signal. It does not reduce the TTFT estimate without measured
  evidence from that worker.

  Returns the normalized candidate augmented with feasibility, derived page and
  token counts, `:estimate/ttft-ms`, and a machine-readable decline reason."
  [request candidate]
  (let [request (protocol/generation-request request)
        candidate (protocol/worker-candidate candidate)
        processed-token-count (dec (count (:request/tokens request)))
        cached-token-count (:candidate/cached-token-count candidate)
        page-size (:candidate/page-size candidate)
        target-token-count (+ processed-token-count
                              (:request/max-new-tokens request))
        resident-pages (if (= :gpu (:candidate/cache-tier candidate))
                         (pages-for cached-token-count page-size)
                         0)
        required-pages (max 0 (- (pages-for target-token-count page-size)
                                 resident-pages))
        available-pages (+ (:candidate/free-pages candidate)
                           (:candidate/evictable-pages candidate))
        missing-token-count (- processed-token-count cached-token-count)
        reason
        (cond
          (not (:candidate/online? candidate)) :worker-offline
          (not (:candidate/model-loaded? candidate)) :model-not-loaded
          (> target-token-count (:candidate/max-context candidate))
          :context-capacity-exceeded
          (neg? missing-token-count) :prefix-longer-than-request
          (> required-pages available-pages) :insufficient-pages
          :else nil)
        ttft-ms
        (when-not reason
          (+ (max (double (:candidate/queue-ms candidate))
                  (double (:candidate/prefix-load-ms candidate)))
             (double (:candidate/gpu-restore-ms candidate))
             (* (double missing-token-count)
                (double (:candidate/prefill-ms-per-token candidate)))
             (double (:candidate/first-token-ms candidate))))]
    (assoc candidate
           :estimate/feasible? (nil? reason)
           :estimate/reason reason
           :estimate/processed-token-count processed-token-count
           :estimate/cached-token-count cached-token-count
           :estimate/missing-token-count missing-token-count
           :estimate/target-token-count target-token-count
           :estimate/resident-pages resident-pages
           :estimate/required-pages required-pages
           :estimate/available-pages available-pages
           :estimate/live-transfer-overlap-eligible?
           (true? (get-in candidate
                          [:candidate/transfer-capabilities
                           :live-overlap-eligible?]))
           :estimate/ttft-ms ttft-ms)))

(defn rank-candidates
  "Return feasible candidates ordered by predicted TTFT and stable tie breaks.

  Lower TTFT wins. Equal estimates prefer more resident prefix tokens and then
  stable worker identity. Every returned value is an `estimate-candidate`
  result; declined candidates are intentionally absent."
  [request candidates]
  (->> candidates
       (mapv #(estimate-candidate request %))
       (filter :estimate/feasible?)
       (sort-by (juxt :estimate/ttft-ms
                      (comp - :estimate/cached-token-count)
                      (comp str :candidate/worker-id)))
       vec))

(defn choose-worker
  "Return the lowest estimated-TTFT feasible candidate, or nil."
  [request candidates]
  (first (rank-candidates request candidates)))

(defn- offer-effect
  [request record candidate]
  {:effect/op :router/send-offer
   :effect/to (:candidate/worker-id candidate)
   :assignment/id (:assignment/id record)
   :assignment/worker-epoch (:candidate/worker-epoch candidate)
   :assignment/request request
   :assignment/candidate candidate})

(defn- cancel-effect
  [request-id assignment]
  {:effect/op :router/send-cancel
   :effect/to (get-in assignment
                      [:assignment/candidate :candidate/worker-id])
   :request/id request-id
   :assignment/id (:assignment/id assignment)})

(defn- offer-next
  [state request-id]
  (let [record (get-in state [:router/requests request-id])
        candidate (first (:assignment/remaining-candidates record))]
    (if candidate
      (let [attempt (inc (long (:assignment/attempt record)))
            assignment-id (protocol/assignment-id request-id attempt)
            next-record (-> record
                            (assoc :assignment/phase :offered
                                   :assignment/attempt attempt
                                   :assignment/id assignment-id
                                   :assignment/candidate candidate)
                            (update :assignment/remaining-candidates subvec 1))]
        {:state (assoc-in state [:router/requests request-id] next-record)
         :effects [(offer-effect (:assignment/request next-record)
                                 next-record candidate)
                   {:effect/op :router/set-offer-timer
                    :request/id request-id
                    :assignment/id assignment-id}]})
      (let [failed (assoc record :assignment/phase :failed
                          :assignment/failure :no-feasible-worker)]
        {:state (assoc-in state [:router/requests request-id] failed)
         :effects [{:effect/op :router/deliver
                    :request/id request-id
                    :response/type :error
                    :response/error :no-feasible-worker}]}))))

(defn transition
  "Advance the cluster router by one event.

  Supported events are `:request/submitted`, `:request/cancelled`,
  `:worker/offer-result`, `:worker/result`, `:worker/unavailable`, and
  `:router/offer-timeout`.
  Every worker completion is fenced by the active assignment identity; stale or
  duplicate events are no-ops. Returns `{:state state' :effects [...]}`."
  [state {:event/keys [type] :as event}]
  (let [state (or state (initial-state))]
    (case type
      :request/submitted
      (let [request (protocol/generation-request (:event/request event))
            request-id (:request/id request)]
        (if (contains? (:router/requests state) request-id)
          {:state state :effects []}
          (let [candidates (rank-candidates request (:event/candidates event))
                record {:assignment/phase :received
                        :assignment/attempt 0
                        :assignment/request request
                        :assignment/remaining-candidates candidates}]
            (offer-next (assoc-in state [:router/requests request-id] record)
                        request-id))))

      :worker/offer-result
      (let [request-id (:request/id event)
            record (get-in state [:router/requests request-id])]
        (if (or (nil? record)
                (not= :offered (:assignment/phase record))
                (not= (:assignment/id record) (:assignment/id event)))
          {:state state :effects []}
          (if (:event/accepted? event)
            (let [assigned (assoc record :assignment/phase :assigned)]
              {:state (assoc-in state [:router/requests request-id] assigned)
               :effects [{:effect/op :router/cancel-offer-timer
                          :request/id request-id
                          :assignment/id (:assignment/id record)}]})
            (offer-next state request-id))))

      :router/offer-timeout
      (let [request-id (:request/id event)
            record (get-in state [:router/requests request-id])]
        (if (and (= :offered (:assignment/phase record))
                 (= (:assignment/id record) (:assignment/id event)))
          (update (offer-next state request-id)
                  :effects #(into [(cancel-effect request-id record)] %))
          {:state state :effects []}))

      :worker/result
      (let [request-id (:request/id event)
            record (get-in state [:router/requests request-id])]
        (if (or (not= :assigned (:assignment/phase record))
                (not= (:assignment/id record) (:assignment/id event)))
          {:state state :effects []}
          (if (= :completed (get-in event [:event/result :status]))
            (let [completed (assoc record :assignment/phase :completed
                                   :assignment/result (:event/result event))]
              {:state (assoc-in state [:router/requests request-id] completed)
               :effects [{:effect/op :router/deliver
                          :request/id request-id
                          :response/type :completed
                          :response/value (:event/result event)}]})
            (offer-next state request-id))))

      :worker/unavailable
      (let [request-id (:request/id event)
            record (get-in state [:router/requests request-id])]
        (if (and (contains? #{:offered :assigned} (:assignment/phase record))
                 (= (:assignment/id record) (:assignment/id event))
                 (= (:worker/id event)
                    (get-in record [:assignment/candidate :candidate/worker-id])))
          (offer-next state request-id)
          {:state state :effects []}))

      :request/cancelled
      (let [request-id (:request/id event)
            record (get-in state [:router/requests request-id])]
        (if (or (nil? record) (contains? terminal-phases (:assignment/phase record)))
          {:state state :effects []}
          {:state (assoc-in state [:router/requests request-id]
                            (assoc record :assignment/phase :cancelled))
           :effects (cond-> [{:effect/op :router/deliver
                              :request/id request-id
                              :response/type :cancelled}]
                      (:assignment/id record)
                      (conj (cancel-effect request-id record)))}))

      (throw (ex-info "Router received an unsupported event" {:event event})))))
