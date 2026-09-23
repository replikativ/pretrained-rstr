(ns pretrained.decision.router
  "Laya checkpoint selection plus lazy, bounded agent residency."
  (:require [clojure.string :as str]
            [pretrained.decision :as decision]
            [pretrained.decision.lang :as lang]))

(def models
  {:english :laya-english
   :multilingual :laya-multilingual
   :typed-decisions :laya-typed-decisions})

(def ^:private aliases
  {:en :english :laya :english :default :english
   :multi :multilingual :ml :multilingual :laya-multilingual :multilingual
   :typed :typed-decisions :typed_decisions :typed-decisions
   :laya-typed-decisions :typed-decisions :decisions :typed-decisions})

(def ^:private typed-workflows
  {:agent-trace-observability #{:action :needs_review :outcome :risk :urgency}
   :customer-service #{:action :category :churn_risk :needs_human :urgency}
   :invoice-processing #{:discrepancy_severity :disposition :duplicate :matches_order :urgency}
   :security-incidents #{:credential_compromise :disposition :severity :true_positive :urgency}})

(declare predict preload! unload!)

(defrecord LayaRouter
    [agents order max-loaded default auto-task-detection? loader lang-guess]
  clojure.lang.IFn
  (invoke [this state questions]
    (predict this state questions))
  (invoke [this state questions opts]
    (predict this state questions opts))
  java.io.Closeable
  (close [this]
    (unload! this)))

(defn normalize-name [name]
  (let [raw (-> name clojure.core/name str/lower-case (str/replace "_" "-") keyword)
        key (get aliases raw raw)]
    (when-not (contains? models key)
      (throw (ex-info (str "unknown Laya model " (pr-str name))
                      {:model name :models (sort (keys models))})))
    key))

(defn match-typed-workflow [questions]
  (let [ids (set (map (comp keyword name) (keys questions)))]
    (some (fn [[workflow signature]] (when (= ids signature) workflow))
          typed-workflows)))

(defn- english-language? [hint]
  (when (some? hint)
    (let [code (-> (if (keyword? hint) (name hint) (str hint)) str/trim str/lower-case
                   (str/split #"\." 2) first
                   (str/replace "_" "-"))
          primary (first (str/split code #"-" 2))]
      (when (seq primary)
        (contains? #{"en" "eng" "english"} primary)))))

(defn- resolve-language-guess [guess state]
  (english-language? (if (fn? guess) (guess state) guess)))

(defn route
  "Select a checkpoint without loading it. Precedence follows upstream Laya:
  model, task, optional workflow detection, language override, caller-supplied
  language guess, text detection. A guess may be a language code or a function
  of state returning a code; nil/blank results fall through to detection."
  ([state questions] (route state questions {}))
  ([state questions {:keys [model task lang lang-guess fallback-lang-guess
                           default auto-task-detection?]
                     :or {default :english}}]
   (let [default (normalize-name default)
         workflow (match-typed-workflow (or questions {}))
         guessed-english? (delay (let [primary (resolve-language-guess lang-guess state)]
                                   (if (some? primary) primary
                                       (resolve-language-guess fallback-lang-guess state))))]
     (cond
       model
       (let [key (normalize-name model)]
         {:model key :checkpoint (models key)
          :reason (str "explicit model=" (pr-str model))
          :detection nil :workflow nil})

       task
       (let [key (normalize-name (if (= "typed-decisions"
                                          (str/replace (str/lower-case (name task)) "_" "-"))
                                   :typed-decisions task))]
         {:model key :checkpoint (models key)
          :reason (str "explicit task=" (pr-str task))
          :detection nil :workflow nil})

       (and workflow auto-task-detection?)
       {:model :typed-decisions :checkpoint (:typed-decisions models)
        :reason (str "question ids match the " (name workflow) " typed-decisions workflow")
        :detection nil :workflow workflow}

       lang
       (let [english? (english-language? (name lang))
             key (if english? :english :multilingual)]
         {:model key :checkpoint (models key)
          :reason (str "explicit lang=" (pr-str lang))
          :detection nil :workflow workflow})

       (some? @guessed-english?)
       (let [key (if @guessed-english?
                   :english :multilingual)]
         {:model key :checkpoint (models key)
          :reason (str "lang-guess identified " (name key) " text")
          :detection nil :workflow workflow})

       :else
       (let [{:keys [script language english? non-latin-fraction] :as detection}
             (lang/analyse state)
             [key reason]
             (cond
               (= script :unknown)
               [default (str "no letters detected in state; using default (" (name default) ")")]

               (not= script :latin)
               [:multilingual
                (format "non-Latin script (%s, %.0f%% of letters); the English checkpoint cannot read it"
                        (name script) (* 100.0 non-latin-fraction))]

               (not english?)
               [:multilingual (str "Latin script but language looks like " (pr-str language)
                                   ", not English")]

               :else [:english "English Latin text"])]
         {:model key :checkpoint (models key) :reason reason
          :detection detection :workflow workflow})))))

(defn router
  "Create a callable Laya router. `(router state questions)` routes and predicts
  in process; the three-argument form accepts route overrides.

  Options: :max-loaded (default 2), :default, :auto-task-detection?,
  :lang-guess, :loader (useful for local checkpoints/tests), and :preload.
  `:lang-guess` is a language code or function of state, overridden per call.
  `:preload` may be true for every
  checkpoint or a collection such as `[:english :multilingual]`."
  ([] (router {}))
  ([{:keys [max-loaded default auto-task-detection? loader preload lang-guess]
     :or {max-loaded 2 default :english loader decision/load-decision}}]
   (let [router (->LayaRouter (atom {}) (atom [])
                              (atom (max 1 (long max-loaded)))
                              (normalize-name default)
                              (boolean auto-task-detection?) loader lang-guess)]
     (cond
       (true? preload) (preload! router)
       (seq preload) (preload! router preload)
       :else router))))

(defn loaded [router]
  (locking router @(:order router)))

(defn- touch! [router key]
  (swap! (:order router) #(conj (vec (remove #{key} %)) key)))

(defn- evict! [router]
  (while (> (count @(:order router)) @(:max-loaded router))
    (let [victim (first @(:order router))]
      (swap! (:order router) #(vec (rest %)))
      (swap! (:agents router) dissoc victim))))

(defn attach!
  "Attach an already loaded agent and retain enough capacity for it."
  [router name agent]
  (locking router
    (let [key (normalize-name name)]
      (swap! (:agents router) assoc key agent)
      (touch! router key)
      (swap! (:max-loaded router) max (count @(:agents router)))
      agent)))

(defn load!
  "Return or lazily load an agent, updating LRU order."
  [router name]
  (locking router
    (let [key (normalize-name name)]
      (if-let [agent (get @(:agents router) key)]
        (do (touch! router key) agent)
        (let [agent ((:loader router) (models key))]
          (swap! (:agents router) assoc key agent)
          (touch! router key)
          (evict! router)
          agent)))))

(defn preload!
  ([router] (preload! router (keys models)))
  ([router names]
   (locking router
     (let [names (mapv normalize-name names)]
       (swap! (:max-loaded router) max (count names) (count @(:agents router)))
       (doseq [name names] (load! router name))
       router))))

(defn unload!
  ([router]
   (locking router
     (reset! (:agents router) {})
     (reset! (:order router) [])
     router))
  ([router name]
   (locking router
     (let [key (normalize-name name)]
       (swap! (:agents router) dissoc key)
       (swap! (:order router) #(vec (remove #{key} %)))
       router))))

(defn predict
  "Route, lazily load, predict, and include the explainable routing decision."
  ([router state questions] (predict router state questions {}))
  ([router state questions opts]
   (let [decision (route state questions
                         (merge {:default (:default router)
                                 :auto-task-detection? (:auto-task-detection? router)
                                 :fallback-lang-guess (:lang-guess router)}
                                opts))
         result (decision/predict (load! router (:model decision)) state questions)]
     (assoc result :routing decision))))
