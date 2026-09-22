(ns pretrained.laya-test
  (:require [clojure.test :refer [deftest is]]
            [pretrained.decision :as decision]
            [pretrained.decision.laya :as laya]))

(deftest python-json-parity
  (is (= "{\"from\": \"José\", \"flags\": [true, null, 3]}"
         (laya/python-json (array-map :from "José" :flags [true nil 3]))))
  (is (= "{\"signature\": \"Trojan:Win32/Emotet\"}"
         (laya/python-json (array-map :signature "Trojan:Win32/Emotet")))))

(deftest loaded-agent-is-an-in-process-function
  (let [agent (laya/map->LayaAgent
               {:dir "/tmp/model" :config {} :encoder {} :tokenizer {}})]
    (with-redefs [laya/predict (fn [actual state questions]
                                {:same-agent? (identical? agent actual)
                                 :state state :questions questions})]
      (is (= {:same-agent? true :state {:body "hello"} :questions {:route {}}}
             (agent {:body "hello"} {:route {}}))))))

(deftest curated-loader-preserves-callability
  (let [agent (laya/map->LayaAgent
               {:dir "/tmp/model" :config {} :encoder {} :tokenizer {}})]
    (with-redefs [laya/load-agent (constantly agent)
                  laya/predict (fn [_ state questions] [state questions])]
      (let [model (decision/load-decision :laya-english "/tmp/model")]
        (is (instance? pretrained.decision.laya.LayaAgent model))
        (is (= [:state :questions] (model :state :questions)))))))

(deftest option-rendering
  (is (= ["billing: invoices, payments" "other"]
         (laya/render-options
          {:type :choice :criteria (array-map :billing "invoices, payments"
                                              :other nil)})))
  (is (= ["level 0: low" "level 1: {\"risk\": \"high\"}"]
         (laya/render-options
          {:type :score :criteria ["low" {:risk "high"}]})))
  (is (= ["false: no, the statement does not hold"
          "true: yes, the statement holds"]
         (laya/render-options {:type :noul})))
  (is (= ["yes" "no"]
         (laya/render-options {:type :choice :criteria ["yes" "no"]})))
  (is (= ["false: false" "true: true"]
         (laya/render-options {:type :noul :criteria {:false false :true true}}))))

(defn- character-tokenizer []
  (let [ids (atom {"[CLS]" 101 "[SEP]" 102 "[MASK]" 103})]
    {:tok {:cls-id 101 :sep-id 102 :mask-id 103 :mask-token "[MASK]"}
     :encode (fn [s]
               (mapv (fn [ch]
                       (or (get @ids (str ch))
                           (get (swap! ids #(if (contains? % (str ch)) %
                                                (assoc % (str ch) (+ 1000 (count %)))))
                                (str ch))))
                     s))}))

(deftest sequence-shape-and-markers
  (let [res (laya/build-sequence
             (character-tokenizer)
             (array-map :body "charged twice")
             {:type :choice
              :instructions "Where?"
              :criteria (array-map :billing "payments" :other nil)}
             {:max-len 80 :head-max-len 48})]
    (is (= 0 (:question-type res)))
    (is (= 101 (first (:ids res))))
    (is (= 102 (last (:ids res))))
    (is (= 2 (count (:markers res))))
    (is (every? #(= 103 (nth (:ids res) %)) (:markers res)))))

(deftest sequence-budget-truncates-options-without-losing-markers
  (let [criteria (into (array-map)
                       (map (fn [i] [(keyword (str "option" i))
                                     (apply str (repeat 30 "x"))])
                            (range 8)))
        res (laya/build-sequence
             (character-tokenizer) "state"
             {:type :choice :instructions "pick" :criteria criteria}
             {:max-len 64 :head-max-len 48})]
    (is (= 8 (count (:markers res))))
    (is (<= (count (:ids res)) 64))
    (is (every? #(= 103 (nth (:ids res) %)) (:markers res)))))

(deftest ^:anchors english-checkpoint-reference-anchor
  (let [dir (str (System/getProperty "user.home")
                 "/.cache/raster/models/convaiinnovations--laya")]
    (if-not (.exists (java.io.File. dir "model.safetensors"))
      (println "SKIP Laya anchor (English checkpoint not present)")
      (let [agent (laya/load-agent dir)
            result (laya/predict
                    agent
                    (array-map :from "user@acme.com"
                               :subject "Duplicate charge"
                               :body "Please refund it.")
                    (array-map
                     :department
                     {:type :choice
                      :instructions "Which department should handle this request?"
                      :criteria (array-map
                                 :billing "invoices, payments, refunds"
                                 :technical "bugs, outages, system errors"
                                 :sales "pricing, new contracts"
                                 :other "everything else")}))
            answer (get-in result [:answers :department])]
        ;; Torch/Laya reference logits:
        ;; [2.1766276 -2.3247652 -3.349228 -2.737616], temp 1.7601519.
        (is (= :billing (:choice answer)))
        (is (= {:billing 0.8459 :technical 0.0656 :sales 0.0366 :other 0.0519}
               (:probabilities answer)))
        (is (= 0.571 (:confidence answer)))
        (is (= {:input-tokens 74 :output-tokens 0} (:usage result)))))))
