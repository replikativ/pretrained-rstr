(ns pretrained.continuation.residency
  "Cost-aware admission and eviction for worker-local GPU continuation pages."
  (:require [pretrained.continuation.page-pool :as page-pool]))

(defn- page-count
  [token-count page-size]
  (if (zero? token-count) 0 (inc (quot (dec token-count) page-size))))

(defn cache-value
  "Estimate the retained value of one resident continuation route.

  Policy fields are read from `:cache/policy`: `:reuse-probability`,
  `:recompute-ms`, `:reload-ms`, `:sharing-bonus`, and `:slo-bonus`. Missing
  values are zero. The result is intentionally in utility milliseconds so
  measured worker observations can replace initial estimates without changing
  the eviction interface."
  [resident-route]
  (let [{:keys [reuse-probability recompute-ms reload-ms sharing-bonus slo-bonus]}
        (:cache/policy resident-route)
        values {:reuse-probability (or reuse-probability 0.0)
                :recompute-ms (or recompute-ms 0.0)
                :reload-ms (or reload-ms 0.0)
                :sharing-bonus (or sharing-bonus 0.0)
                :slo-bonus (or slo-bonus 0.0)}]
    (doseq [[field value] values]
      (when-not (and (number? value) (not (neg? (double value))))
        (throw (ex-info "Route policy costs must be non-negative numbers"
                        {:continuation-id (:continuation-id resident-route)
                         :field field :value value}))))
    (let [probability (double (:reuse-probability values))]
      (when-not (<= 0.0 probability 1.0)
        (throw (ex-info "Route reuse probability must be in [0,1]"
                        {:continuation-id (:continuation-id resident-route)
                         :reuse-probability probability})))
      (+ (* probability
            (max 0.0 (- (double (:recompute-ms values))
                        (double (:reload-ms values)))))
         (double (:sharing-bonus values))
         (double (:slo-bonus values))))))

(defn- value-density
  [resident-route]
  (/ (cache-value resident-route)
     (double (max 1 (count (:pages resident-route))))))

(defn- leased-continuations
  [snapshot]
  (into #{} (mapcat :continuation-ids) (vals (:leases snapshot))))

(defn- eligible-route?
  [leased protected resident-route]
  (let [policy (:cache/policy resident-route)]
    (and (true? (:durable? policy))
         (not (:pinned? policy))
         (not (:pending resident-route))
         (not (contains? leased (:continuation-id resident-route)))
         (not (contains? protected (:continuation-id resident-route))))))

(defn- release-simulated-route
  [simulation resident-route]
  (reduce (fn [current page]
            (let [refs (get-in current [:refcounts page] 0)]
              (if (= refs 1)
                (-> current
                    (update :refcounts dissoc page)
                    (update :free-pages conj page))
                (update-in current [:refcounts page] dec))))
          simulation
          (:pages resident-route)))

(defn- plan-pages
  [snapshot required protected-continuation-ids]
  (let [available (count (:free-pages snapshot))
        leased (leased-continuations snapshot)
        candidates
        (sort-by (juxt value-density
                       #(long (or (get-in % [:cache/policy :last-access]) 0))
                       (comp str :continuation-id))
                 (filter #(eligible-route? leased protected-continuation-ids %)
                         (vals (:routes snapshot))))]
    (loop [remaining candidates
           simulation (select-keys snapshot [:refcounts :free-pages])
           evictions []]
      (let [free (count (:free-pages simulation))]
        (cond
          (>= free required)
          {:admissible? true
           :required-pages required
           :available-pages available
           :evictions evictions
           :free-pages-after-eviction free}

          (empty? remaining)
          {:admissible? false
           :required-pages required
           :available-pages available
           :evictions []
           :shortfall-pages (- required free)}

          :else
          (let [victim (first remaining)]
            (recur (next remaining)
                   (release-simulated-route simulation victim)
                   (conj evictions (:continuation-id victim)))))))))

(defn plan-admission
  "Plan durable route evictions needed to admit `token-count` tokens.

  `opts` may contain `:protected-continuation-ids`. Only durable, unpinned and
  unleased routes are candidates. Lowest value is evicted first, with oldest
  `:last-access` and stable continuation identity as tie breakers. Shared-page
  reference counts are simulated, so the plan accounts for evictions that free
  no page until another sharing route is also removed."
  ([snapshot token-count] (plan-admission snapshot token-count {}))
  ([snapshot token-count {:keys [protected-continuation-ids]
                          :or {protected-continuation-ids #{}}}]
   (when-not (and (integer? token-count) (not (neg? token-count)))
     (throw (ex-info "Admission token count must be a non-negative integer"
                     {:token-count token-count})))
   (plan-pages snapshot
               (page-count token-count (:page-size snapshot))
               protected-continuation-ids)))

(defn plan-capacity-admission
  "Plan evictions for a continuation's incremental projected capacity.

  Existing resident pages are credited and a shared partial tail's possible
  copy-on-write page is included. The target continuation is always protected
  from eviction. `opts` accepts `:protected-continuation-ids`."
  ([snapshot continuation-id token-capacity]
   (plan-capacity-admission snapshot continuation-id token-capacity {}))
  ([snapshot continuation-id token-capacity
    {:keys [protected-continuation-ids]
     :or {protected-continuation-ids #{}}}]
   (plan-pages
    snapshot
    (page-pool/capacity-page-requirement
     snapshot continuation-id token-capacity)
    (conj protected-continuation-ids continuation-id))))

(defn evictable-page-count
  "Return unique physical pages recoverable by evicting every eligible route.

  `opts` accepts `:protected-continuation-ids`. Shared refcounts are simulated,
  so two eligible forks can jointly contribute a page while evicting only one
  of them may contribute none."
  ([snapshot] (evictable-page-count snapshot {}))
  ([snapshot {:keys [protected-continuation-ids]
              :or {protected-continuation-ids #{}}}]
   (let [leased (leased-continuations snapshot)
         eligible (filter #(eligible-route? leased protected-continuation-ids %)
                          (vals (:routes snapshot)))
         simulation (reduce release-simulated-route
                            (select-keys snapshot [:refcounts :free-pages])
                            eligible)]
     (- (count (:free-pages simulation))
        (count (:free-pages snapshot))))))

(defn reserve-admission!
  "Evict eligible routes and reserve projected capacity atomically.

  Returns an admission plan with `:capacity-reservation` when admissible, or an
  unchanged non-admissible plan. The reservation must be released explicitly
  after completion, cancellation, or failed restore."
  ([pool continuation-id token-capacity]
   (reserve-admission! pool continuation-id token-capacity {}))
  ([pool continuation-id token-capacity opts]
   (locking pool
     (let [plan (plan-capacity-admission
                 (page-pool/residency-snapshot pool)
                 continuation-id token-capacity opts)]
       (if-not (:admissible? plan)
         plan
         (do
           (doseq [victim (:evictions plan)]
             (page-pool/release-route! pool victim))
           (assoc plan :capacity-reservation
                  (page-pool/reserve-capacity!
                   pool continuation-id token-capacity))))))))

(defn admit-route!
  "Safely evict durable routes and allocate a new resident continuation route.

  Planning and application are serialized on the pool, so a lease cannot appear
  between validation and eviction. Returns the plan plus `:resident-route`, or
  an unchanged non-admissible plan. `opts` accepts `:start-position`, `:policy`,
  and `:protected-continuation-ids`."
  ([pool continuation-id token-count]
   (admit-route! pool continuation-id token-count {}))
  ([pool continuation-id token-count opts]
   (locking pool
     (let [plan (plan-admission
                 (page-pool/residency-snapshot pool) token-count opts)]
       (if-not (:admissible? plan)
         plan
         (do
           (doseq [victim (:evictions plan)]
             (page-pool/release-route! pool victim))
           (assoc plan :resident-route
                  (page-pool/allocate-route!
                   pool continuation-id token-count
                   (select-keys opts [:start-position :policy])))))))))
