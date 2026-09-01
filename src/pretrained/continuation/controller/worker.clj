(ns pretrained.continuation.controller.worker
  "Pure device-local continuation lifecycle and resource reservations.

  This machine is authoritative for admission. Router candidates are only
  observations and may be stale; an offer is accepted only after the worker
  rechecks its current epoch, model set, and page capacity. Storage and Raster
  calls are emitted as effects and fenced by assignment identity."
  (:require [pretrained.continuation.controller.protocol :as protocol]
            [pretrained.continuation.controller.router :as router]))

(def terminal-phases #{:completed :failed :cancelled})

(defn initial-state
  "Create a worker state.

  Options require `:worker/id`, non-negative `:worker/epoch`, a set of
  `:worker/models`, and non-negative `:worker/free-pages` and
  `:worker/evictable-pages`. `:worker/online?` defaults to true."
  [{:worker/keys [id epoch models free-pages evictable-pages online?] :as opts}]
  (when (nil? id)
    (throw (ex-info "Controller worker requires :worker/id" {:opts opts})))
  (when-not (and (integer? epoch) (not (neg? epoch)))
    (throw (ex-info "Controller worker requires a non-negative epoch" {:opts opts})))
  (when-not (set? models)
    (throw (ex-info "Controller worker models must be a set" {:opts opts})))
  (doseq [[field value] [[:worker/free-pages free-pages]
                         [:worker/evictable-pages evictable-pages]]]
    (when-not (and (integer? value) (not (neg? value)))
      (throw (ex-info "Controller worker page counts must be non-negative integers"
                      {:field field :value value}))))
  {:worker/id id
   :worker/epoch (long epoch)
   :worker/models models
   :worker/free-pages (long free-pages)
   :worker/evictable-pages (long evictable-pages)
   :worker/online? (not (false? online?))
   :worker/assignments {}
   :worker/resident-routes {}})

(defn- offer-result
  [state assignment accepted? reason]
  {:effect/op :worker/send-offer-result
   :effect/to :router
   :request/id (get-in assignment [:assignment/request :request/id])
   :assignment/id (:assignment/id assignment)
   :assignment/worker-epoch (:worker/epoch state)
   :event/accepted? accepted?
   :event/reason reason})

(defn- result-effect
  [assignment result]
  {:effect/op :worker/send-result
   :effect/to :router
   :request/id (get-in assignment [:assignment/request :request/id])
   :assignment/id (:assignment/id assignment)
   :event/result result})

(defn- operation-effect
  [assignment op]
  {:effect/op op
   :worker/id (:assignment/worker-id assignment)
   :request/id (get-in assignment [:assignment/request :request/id])
   :assignment/id (:assignment/id assignment)
   :assignment/request (:assignment/request assignment)
   :assignment/candidate (:assignment/candidate assignment)})

(defn- next-operation
  [assignment]
  (let [cached (get-in assignment [:assignment/candidate
                                   :estimate/cached-token-count])
        missing (get-in assignment [:assignment/candidate
                                    :estimate/missing-token-count])
        tier (get-in assignment [:assignment/candidate :candidate/cache-tier])]
    (cond
      (and (pos? cached) (not= :gpu tier))
      [(assoc assignment :assignment/phase :restoring)
       (operation-effect assignment :worker/restore-prefix)]

      (pos? missing)
      [(assoc assignment :assignment/phase :prefilling)
       (operation-effect assignment :worker/prefill-suffix)]

      :else
      [(assoc assignment :assignment/phase :decoding)
       (operation-effect assignment :worker/decode)])))

(defn- release-reservation
  [state assignment]
  (let [free-reserved (:assignment/free-pages-reserved assignment 0)
        evicted (:assignment/evicted-pages assignment 0)]
    (-> state
        (update :worker/free-pages + free-reserved evicted)
        (update :worker/assignments dissoc (:assignment/id assignment)))))

(defn- fail-assignment
  [state assignment reason]
  {:state (release-reservation state assignment)
   :effects [(result-effect assignment {:status :failed :reason reason})]})

(defn- active-assignment
  [state event]
  (get-in state [:worker/assignments (:assignment/id event)]))

(defn transition
  "Advance a device-local worker by one controller event.

  Supported events are `:assignment/offered`, operation result events for
  restore, prefill, and decode, `:assignment/cancelled`, and lifecycle
  `:worker/crashed`/`:worker/restarted`. Operation results for stale assignment
  identities are ignored. Returns `{:state state' :effects [...]}`."
  [state {:event/keys [type] :as event}]
  (case type
    :assignment/offered
    (let [request (protocol/generation-request (:assignment/request event))
          candidate (router/estimate-candidate request (:assignment/candidate event))
          assignment-id (:assignment/id event)
          request-id (:request/id request)
          duplicate (get-in state [:worker/assignments assignment-id])
          request-active?
          (some (fn [[id assignment]]
                  (and (not= id assignment-id)
                       (= request-id (get-in assignment
                                             [:assignment/request :request/id]))
                       (not (contains? terminal-phases
                                       (:assignment/phase assignment)))))
                (:worker/assignments state))
          required (:estimate/required-pages candidate)
          free (:worker/free-pages state)
          evictable (:worker/evictable-pages state)
          reason
          (cond
            duplicate nil
            (not (:worker/online? state)) :worker-offline
            (not= (:assignment/worker-epoch event) (:worker/epoch state))
            :worker-epoch-changed
            (not= (:candidate/worker-id candidate) (:worker/id state))
            :wrong-worker
            (not (contains? (:worker/models state)
                            (:request/model-fingerprint request)))
            :model-not-loaded
            request-active? :request-already-assigned
            (> required (+ free evictable)) :insufficient-pages
            :else nil)]
      (cond
        duplicate
        {:state state
         :effects (cond-> [(offer-result state duplicate true nil)]
                    (= :completed (:assignment/phase duplicate))
                    (conj (result-effect duplicate (:assignment/result duplicate))))}

        reason
        {:state state
         :effects [(offer-result state
                                 {:assignment/id assignment-id
                                  :assignment/request request}
                                 false reason)]}

        :else
        (let [from-free (min free required)
              from-eviction (- required from-free)
              assignment {:assignment/id assignment-id
                          :assignment/worker-id (:worker/id state)
                          :assignment/request request
                          :assignment/candidate candidate
                          :assignment/free-pages-reserved from-free
                          :assignment/evicted-pages from-eviction
                          :assignment/phase :accepted}
              [assignment operation] (next-operation assignment)]
          {:state (-> state
                      (update :worker/free-pages - from-free)
                      (update :worker/evictable-pages - from-eviction)
                      (assoc-in [:worker/assignments assignment-id] assignment))
           :effects [(offer-result state assignment true nil) operation]})))

    :worker/restore-result
    (if-let [assignment (active-assignment state event)]
      (if-not (:event/ok? event)
        (fail-assignment state assignment
                         (or (:event/reason event) :restore-failed))
        (let [missing (get-in assignment [:assignment/candidate
                                          :estimate/missing-token-count])
              [assignment operation]
              (if (pos? missing)
                [(assoc assignment :assignment/phase :prefilling)
                 (operation-effect assignment :worker/prefill-suffix)]
                [(assoc assignment :assignment/phase :decoding)
                 (operation-effect assignment :worker/decode)])]
          {:state (assoc-in state [:worker/assignments (:assignment/id event)]
                            assignment)
           :effects [operation]}))
      {:state state :effects []})

    :worker/prefill-result
    (if-let [assignment (active-assignment state event)]
      (if-not (:event/ok? event)
        (fail-assignment state assignment
                         (or (:event/reason event) :prefill-failed))
        (let [assignment (assoc assignment :assignment/phase :decoding)]
          {:state (assoc-in state [:worker/assignments (:assignment/id event)]
                            assignment)
           :effects [(operation-effect assignment :worker/decode)]}))
      {:state state :effects []})

    :worker/decode-result
    (if-let [assignment (active-assignment state event)]
      (if-not (:event/ok? event)
        (fail-assignment state assignment
                         (or (:event/reason event) :decode-failed))
        (let [request-id (get-in assignment [:assignment/request :request/id])
              result {:status :completed :tokens (:event/tokens event)}
              completed (assoc assignment
                               :assignment/phase :completed
                               :assignment/result result)
              state (-> state
                        (assoc-in [:worker/assignments (:assignment/id event)] completed)
                        (assoc-in [:worker/resident-routes request-id]
                                  {:request/id request-id
                                   :assignment/id (:assignment/id assignment)
                                   :pages (+ (:assignment/free-pages-reserved assignment)
                                             (:assignment/evicted-pages assignment))}))]
          {:state state
           :effects [(result-effect completed result)]}))
      {:state state :effects []})

    :assignment/cancelled
    (if-let [assignment (active-assignment state event)]
      (if (contains? terminal-phases (:assignment/phase assignment))
        {:state state :effects []}
        {:state (release-reservation state assignment)
         :effects [{:effect/op :worker/cancel-operation
                    :request/id (get-in assignment
                                        [:assignment/request :request/id])
                    :assignment/id (:assignment/id assignment)}]})
      {:state state :effects []})

    :worker/crashed
    (let [active-pages
          (reduce + 0
                  (map (fn [[_ assignment]]
                         (if (contains? terminal-phases (:assignment/phase assignment))
                           0
                           (+ (:assignment/free-pages-reserved assignment 0)
                              (:assignment/evicted-pages assignment 0))))
                       (:worker/assignments state)))
          resident-pages (reduce + 0 (map :pages (vals (:worker/resident-routes state))))]
      {:state (-> state
                (assoc :worker/online? false
                       :worker/assignments {}
                       :worker/resident-routes {}
                       :worker/free-pages
                       (+ (:worker/free-pages state)
                          active-pages
                          resident-pages))
                (update :worker/epoch inc))
       :effects []})

    :worker/restarted
    {:state (assoc state :worker/online? true) :effects []}

    (throw (ex-info "Controller worker received an unsupported event"
                    {:event event}))))
