(ns pretrained.laya-router-test
  (:require [clojure.test :refer [deftest is testing]]
            [pretrained.decision :as decision]
            [pretrained.decision.lang :as lang]
            [pretrained.decision.router :as router]))

(deftest script-and-language-detection
  (is (= :devanagari (:script (lang/analyse {:body "मुझसे दो बार शुल्क लिया गया"}))))
  (is (= :kana (lang/detect-script "請求が重複しています")))
  (is (= :en (lang/guess-latin-language "Please refund the charge on this account")))
  (is (= :de (lang/guess-latin-language "Das ist nicht richtig und die Zahlung ist doppelt"))))

(deftest route-precedence-and-explanations
  (let [questions {:department {:type :choice}}]
    (is (= :english (:model (router/route "English request" questions))))
    (is (= :multilingual (:model (router/route "मुझसे दो बार शुल्क लिया गया" questions))))
    (is (= :english (:model (router/route "हिन्दी" questions {:model :english}))))
    (is (= :multilingual (:model (router/route "hello" questions {:lang :de}))))))

(deftest language-guess-precedence-and-abstention
  (let [questions {:department {:type :choice}}
        calls (atom 0)
        guess (fn [_] (swap! calls inc) "pt_BR.UTF-8")]
    (is (= :multilingual (:model (router/route "Please refund this" questions
                                               {:lang-guess guess}))))
    (is (= 1 @calls))
    (is (= :english (:model (router/route "Please refund this" questions
                                          {:lang-guess :en_US}))))
    (is (= :multilingual (:model (router/route "Please refund this" questions
                                               {:lang-guess (constantly nil)
                                                :fallback-lang-guess "fr"}))))
    (is (= :english (:model (router/route "Please refund this" questions
                                          {:lang-guess (constantly " ")}))))
    (is (= :english (:model (router/route "Please refund this" questions
                                          {:model :english :lang-guess guess}))))
    (is (= 1 @calls))
    (let [r (router/router {:lang-guess "de"
                            :loader (constantly {:checkpoint :laya-multilingual})})]
      (with-redefs [decision/predict (fn [_ _ _] {})]
        (is (= :multilingual (get-in (r "Please refund this" questions)
                                     [:routing :model])))
        (is (= :english (get-in (r "Please refund this" questions
                                  {:lang-guess "en-US"}) [:routing :model])))))))

(deftest typed-workflow-detection-is-exact-and-opt-in
  (let [questions (zipmap [:action :needs_review :outcome :risk :urgency] (repeat {}))]
    (is (= :agent-trace-observability (router/match-typed-workflow questions)))
    (is (= :english (:model (router/route "English state" questions))))
    (is (= :typed-decisions
           (:model (router/route "English state" questions {:auto-task-detection? true}))))))

(deftest lazy-loading-attachment-and-lru
  (let [loads (atom [])
        r (router/router {:max-loaded 1
                          :loader (fn [checkpoint]
                                    (swap! loads conj checkpoint)
                                    {:checkpoint checkpoint})})]
    (is (= {:checkpoint :laya-english} (router/load! r :english)))
    (is (= [:english] (router/loaded r)))
    (router/load! r :multilingual)
    (is (= [:multilingual] (router/loaded r)))
    (is (= [:laya-english :laya-multilingual] @loads))
    (router/attach! r :english {:attached true})
    (is (= #{:english :multilingual} (set (router/loaded r))))
    (router/unload! r :multilingual)
    (is (= [:english] (router/loaded r)))))

(deftest router-is-callable-and-can-preload
  (let [loads (atom [])
        r (router/router {:preload [:english :multilingual]
                          :loader (fn [checkpoint]
                                    (swap! loads conj checkpoint)
                                    {:checkpoint checkpoint})})]
    (is (= [:laya-english :laya-multilingual] @loads))
    (with-redefs [decision/predict
                  (fn [agent state questions]
                    {:checkpoint (:checkpoint agent)
                     :state state :questions questions})]
      (let [result (r {:body "Please refund this"} {:department {:type :choice}})]
        (is (= :laya-english (:checkpoint result)))
        (is (= :english (get-in result [:routing :model])))))
    (.close ^java.io.Closeable r)
    (is (empty? (router/loaded r)))))

(deftest default-router-keeps-two-language-checkpoints
  (let [r (router/router {:loader (fn [checkpoint] {:checkpoint checkpoint})})]
    (router/load! r :english)
    (router/load! r :multilingual)
    (is (= [:english :multilingual] (router/loaded r)))))

(deftest concurrent-loads-share-an-agent
  (let [loads (atom 0)
        start (promise)
        r (router/router {:loader (fn [checkpoint]
                                    (swap! loads inc)
                                    (Thread/sleep 25)
                                    {:checkpoint checkpoint})})
        workers (doall (repeatedly 8 #(future @start (router/load! r :english))))]
    (deliver start true)
    (is (apply = (map deref workers)))
    (is (= 1 @loads))
    (is (= [:english] (router/loaded r)))))

(deftest gpu-router-options-close-evicted-and-unloaded-agents
  (let [loads (atom [])
        closed (atom [])]
    (with-redefs [decision/load-decision
                  (fn [checkpoint opts]
                    (swap! loads conj [checkpoint opts])
                    (reify java.io.Closeable
                      (close [_] (swap! closed conj checkpoint))))]
      (let [r (router/router {:max-loaded 1
                              :decision-opts {:gpu? true :target :ze:0}})]
        (router/load! r :english)
        (router/load! r :multilingual)
        (is (= [:laya-english] @closed))
        (.close ^java.io.Closeable r)
        (is (= [:laya-english :laya-multilingual] @closed))
        (is (= [[:laya-english {:gpu? true :target :ze:0}]
                [:laya-multilingual {:gpu? true :target :ze:0}]]
               @loads))))))
