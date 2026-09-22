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
