(ns pretrained.decision.laya
  "Laya typed-decision input contract and checkpoint metadata.

  Prompt construction is independent of model execution so tokenizer and
  marker parity can be established before numerical encoder work, and the same
  contract can serve CPU and resident GPU backends."
  (:require [clojure.data.json :as json]
            [clojure.string :as str]
            [raster.dl.array-ops :as ops]
            [raster.dl.nn :as nn]
            [pretrained.arch.modernbert :as modernbert]
            [pretrained.tokenizer.bpe :as bpe]
            [pretrained.tokenizer.sp :as sp]))

(def question-types {"choice" 0, "score" 1, "noul" 2})

(declare predict)

(defrecord LayaAgent [dir config encoder tokenizer]
  clojure.lang.IFn
  (invoke [this state questions]
    (predict this state questions)))

(def checkpoints
  {:laya-english
   {:hf "convaiinnovations/laya"
    :snapshot :laya-english
    :include ["model.safetensors" "rl_agent_config.json"
              "encoder/config.json" "tokenizer/tokenizer.json"
              "tokenizer/tokenizer_config.json"]}
   :laya-multilingual
   {:hf "convaiinnovations/laya"
    :snapshot :laya-multilingual
    :subfolder "multilingual"
    :include ["multilingual/**"]}
   :laya-typed-decisions
   {:hf "convaiinnovations/laya"
    :snapshot :laya-typed-decisions
    :subfolder "typed-decisions"
    :include ["typed-decisions/**"]}})

(defn- getk [m k]
  (if (contains? m k) (get m k) (get m (keyword k))))

(defn- hask? [m k]
  (or (contains? m k) (contains? m (keyword k))))

(declare python-json)

(defn- json-key [x]
  (json/write-str (if (keyword? x) (name x) (str x))
                  :escape-unicode false :escape-slash false))

(defn python-json
  "Serialize the JSON subset used by Laya like Python json.dumps defaults:
  Unicode remains literal and separators contain one following space."
  [x]
  (cond
    (map? x) (str "{" (str/join ", " (map (fn [[k v]]
                                               (str (json-key k) ": " (python-json v))) x)) "}")
    (sequential? x) (str "[" (str/join ", " (map python-json x)) "]")
    (keyword? x) (json/write-str (name x) :escape-unicode false)
    :else (json/write-str x :escape-unicode false :escape-slash false)))

(defn serialize-state [state]
  (if (string? state) state (python-json state)))

(defn- rendered [x]
  (if (string? x) x (python-json x)))

(defn- score-criteria [criteria]
  (if (map? criteria)
    (mapv #(if (keyword? %) (name %) %) (keys criteria))
    criteria))

(defn render-options
  "Render external or internal question definitions in stable option order."
  [question]
  (let [type (name (getk question "type"))
        criteria (getk question "criteria")]
    (case type
      "choice"
      (mapv (fn [[label description]]
              (let [label (name label)]
                (if (or (nil? description) (= "" description))
                  label
                  (str label ": " (rendered description)))))
            (if (map? criteria) criteria (map vector criteria (repeat nil))))
      "score"
      (mapv (fn [i criterion] (str "level " i ": " (rendered criterion)))
            (range) (score-criteria criteria))
      "noul"
      (let [criteria (or criteria {})
            f (getk criteria "false")
            t (getk criteria "true")]
        [(str "false: " (if (or (nil? f) (= "" f))
                          "no, the statement does not hold" (rendered f)))
         (str "true: " (if (or (nil? t) (= "" t))
                         "yes, the statement holds" (rendered t)))])
      (throw (ex-info (str "unknown Laya question type " (pr-str type))
                      {:question question :type type})))))

(defn load-tokenizer
  "Load the checkpoint-local tokenizer with an explicit no-special encoder."
  [dir]
  (let [path (str dir "/tokenizer/tokenizer.json")
        raw (json/read-str (slurp path))
        model (get raw "model")]
    (if (get model "byte_fallback")
      (let [tok (sp/load-tokenizer path)]
        {:kind :sp :tok tok :encode #(sp/encode tok % false)})
      (let [tok (bpe/load-bpe-tokenizer path)]
        {:kind :bpe :tok tok :encode #(vec (bpe/encode tok %))}))))

(defn build-sequence
  "Build one Laya sequence and option-marker vector.

  `tokenizer` is the value from load-tokenizer. Options accept :max-len,
  :head-max-len, and :truncate-left? and match the upstream defaults."
  ([tokenizer state question] (build-sequence tokenizer state question {}))
  ([{:keys [tok encode]} state question
    {:keys [max-len head-max-len truncate-left?]
     :or {max-len 512 head-max-len 192 truncate-left? false}}]
   (let [type (name (getk question "type"))
         instructions (let [x (getk question "instructions")]
                        (if (string? x) x (python-json x)))
         mask-token (:mask-token tok)
         clean #(str/replace (str %) mask-token " ")
         head-ids (vec (encode (str type " question: " (clean instructions))))
         option-ids (mapv #(into [(:mask-id tok)]
                                (take 48 (encode (str " " (clean %)))))
                          (render-options question))
         option-budget (- head-max-len (reduce + (map count option-ids)))
         per (when (< option-budget 16)
               (max 4 (quot (- head-max-len 16) (max 1 (count option-ids)))))
         option-ids (if per (mapv #(vec (take per %)) option-ids) option-ids)
         option-budget (- head-max-len (reduce + (map count option-ids)))
         head-ids (vec (take (max 8 option-budget) head-ids))
         prefix (into [(:cls-id tok)] (concat head-ids [(:sep-id tok)]))
         [prefix markers]
         (reduce (fn [[ids markers] option]
                   [(into ids option) (conj markers (count ids))])
                 [prefix []] option-ids)
         prefix (conj prefix (:sep-id tok))
         room (max 0 (- max-len (count prefix) 1))
         state-ids (vec (encode (clean (serialize-state state))))
         state-ids (if truncate-left?
                     (vec (take-last room state-ids))
                     (vec (take room state-ids)))
         ids (vec (take max-len (concat prefix state-ids [(:sep-id tok)])))]
     {:ids ids
      :markers (vec (filter #(< % max-len) markers))
      :question-type (get question-types type)})))

;; ---------------------------------------------------------------------------
;; Decision head and public prediction path
;; ---------------------------------------------------------------------------

(defn load-agent
  "Load one callable local Laya checkpoint. The returned agent supports both
  `(predict agent state questions)` and direct `(agent state questions)` use.
  The same F32 weight map backs the encoder and decision head, avoiding a
  second expansion of the checkpoint."
  [dir]
  (let [config (json/read-str (slurp (str dir "/rl_agent_config.json")))
        encoder (modernbert/load-model dir)]
    (->LayaAgent dir config encoder (load-tokenizer dir))))

(defn- weight ^floats [agent name]
  (modernbert/weight (:encoder agent) name))

(defn- residual ^floats [^floats a ^floats b]
  (modernbert/residual-add a b))

(defn- add-question-type
  ^floats [agent ^floats hidden seq-len question-type]
  (let [d (:d-model (:encoder agent))
        ^floats type-emb (weight agent "type_emb.weight")
        out (aclone hidden)
        type-base (* (long question-type) d)]
    (dotimes [row seq-len]
      (dotimes [col d]
        (let [i (+ (* row d) col)]
          (aset out i (+ (aget out i) (aget type-emb (+ type-base col)))))))
    out))

(defn- full-attention
  ^floats [agent ^floats x seq-len layer]
  (let [d (:d-model (:encoder agent))
        heads (quot d 64)
        prefix (str "head.layers." layer ".")
        qkv (nn/linear x
                       (weight agent (str prefix "self_attn.in_proj_weight"))
                       (weight agent (str prefix "self_attn.in_proj_bias"))
                       seq-len d (* 3 d))
        q (ops/slice-strided-2d qkv seq-len (* 3 d) 0 d)
        k (ops/slice-strided-2d qkv seq-len (* 3 d) d d)
        v (ops/slice-strided-2d qkv seq-len (* 3 d) (* 2 d) d)
        scores (float-array (* seq-len heads seq-len))
        context (float-array (* seq-len d))
        _ (modernbert/attention! q k v scores context seq-len heads 64 (/ 1.0 8.0) 0)
        projected (nn/linear context
                             (weight agent (str prefix "self_attn.out_proj.weight"))
                             (weight agent (str prefix "self_attn.out_proj.bias"))
                             seq-len d d)]
    projected))

(defn- head-layer
  ^floats [agent ^floats x seq-len layer]
  (let [d (:d-model (:encoder agent))
        ffn (* 4 d)
        prefix (str "head.layers." layer ".")
        n1 (modernbert/layer-norm x
                                  (weight agent (str prefix "norm1.weight"))
                                  (weight agent (str prefix "norm1.bias"))
                                  seq-len d 1.0e-5)
        x1 (residual x (full-attention agent n1 seq-len layer))
        n2 (modernbert/layer-norm x1
                                  (weight agent (str prefix "norm2.weight"))
                                  (weight agent (str prefix "norm2.bias"))
                                  seq-len d 1.0e-5)
        up (nn/linear n2
                      (weight agent (str prefix "linear1.weight"))
                      (weight agent (str prefix "linear1.bias"))
                      seq-len d ffn)
        ;; torch.nn.TransformerEncoderLayer defaults to ReLU. Laya does not
        ;; override `activation` when constructing its two head layers.
        activated (float-array (* seq-len ffn))
        _ (modernbert/relu! up activated (* seq-len ffn))
        down (nn/linear activated
                        (weight agent (str prefix "linear2.weight"))
                        (weight agent (str prefix "linear2.bias"))
                        seq-len ffn d)]
    (residual x1 down)))

(defn- decision-hidden
  ^floats [agent token-ids question-type]
  (let [seq-len (count token-ids)
        encoded (modernbert/encode (:encoder agent) token-ids)
        typed (add-question-type agent encoded seq-len question-type)
        layers (long (get (:config agent) "head_layers" 2))]
    (loop [x typed layer 0]
      (if (< layer layers)
        (recur (head-layer agent x seq-len layer) (inc layer))
        x))))

(defn- copy-rows
  ^floats [^floats src row0 rows width]
  (let [out (float-array (* rows width))]
    (System/arraycopy src (* row0 width) out 0 (* rows width))
    out))

(defn- add-question-types-batch
  ^floats [agent ^floats hidden batch max-len question-types]
  (let [d (:d-model (:encoder agent))
        ^floats type-emb (weight agent "type_emb.weight")
        out (aclone hidden)]
    (doseq [b (range batch)
            row (range max-len)
            col (range d)]
      (let [i (+ (* (+ (* b max-len) row) d) col)
            type-i (+ (* (long (nth question-types b)) d) col)]
        (aset out i (+ (aget out i) (aget type-emb type-i)))))
    out))

(defn- full-attention-batch
  ^floats [agent ^floats x batch max-len lengths layer]
  (let [d (:d-model (:encoder agent))
        heads (quot d 64)
        rows (* batch max-len)
        prefix (str "head.layers." layer ".")
        qkv (nn/linear x
                       (weight agent (str prefix "self_attn.in_proj_weight"))
                       (weight agent (str prefix "self_attn.in_proj_bias"))
                       rows d (* 3 d))
        context (float-array (* rows d))]
    (doseq [b (range batch)]
      (let [len (long (nth lengths b))
            row0 (* b max-len)
            source-row0 (* row0 (* 3 d))
            scores (float-array (* len heads len))
            out (float-array (* len d))
            _ (modernbert/attention-strided!
               qkv qkv qkv scores out len heads 64 (/ 1.0 8.0) 0
               (* 3 d) source-row0
               (* 3 d) (+ source-row0 d)
               (* 3 d) (+ source-row0 (* 2 d)))]
        (System/arraycopy out 0 context (* row0 d) (* len d))))
    (nn/linear context
               (weight agent (str prefix "self_attn.out_proj.weight"))
               (weight agent (str prefix "self_attn.out_proj.bias"))
               rows d d)))

(defn- head-layer-batch
  ^floats [agent ^floats x batch max-len lengths layer]
  (let [d (:d-model (:encoder agent))
        rows (* batch max-len)
        ffn (* 4 d)
        prefix (str "head.layers." layer ".")
        n1 (modernbert/layer-norm x (weight agent (str prefix "norm1.weight"))
                                  (weight agent (str prefix "norm1.bias")) rows d 1.0e-5)
        x1 (residual x (full-attention-batch agent n1 batch max-len lengths layer))
        n2 (modernbert/layer-norm x1 (weight agent (str prefix "norm2.weight"))
                                  (weight agent (str prefix "norm2.bias")) rows d 1.0e-5)
        up (nn/linear n2 (weight agent (str prefix "linear1.weight"))
                      (weight agent (str prefix "linear1.bias")) rows d ffn)
        activated (float-array (* rows ffn))
        _ (modernbert/relu! up activated (* rows ffn))
        down (nn/linear activated (weight agent (str prefix "linear2.weight"))
                        (weight agent (str prefix "linear2.bias")) rows ffn d)]
    (residual x1 down)))

(defn- decision-hidden-batch
  [agent items]
  (let [{:keys [data lengths max-len batch]}
        (modernbert/encode-batch (:encoder agent) (mapv :ids items))
        typed (add-question-types-batch agent data batch max-len
                                        (mapv :question-type items))
        layers (long (get (:config agent) "head_layers" 2))
        hidden (loop [x typed layer 0]
                 (if (< layer layers)
                   (recur (head-layer-batch agent x batch max-len lengths layer)
                          (inc layer))
                   x))]
    {:data hidden :lengths lengths :max-len max-len :batch batch}))

(defn- gather-marker-rows
  ^floats [^floats hidden markers d]
  (let [out (float-array (* (count markers) d))]
    (doseq [[row marker] (map-indexed vector markers)]
      (System/arraycopy hidden (* (long marker) d) out (* (long row) d) d))
    out))

(defn- score-markers
  ^floats [agent ^floats hidden markers]
  (let [d (:d-model (:encoder agent))
        n (count markers)
        marked (gather-marker-rows hidden markers d)
        normalized (modernbert/layer-norm marked (weight agent "scorer.0.weight")
                                          (weight agent "scorer.0.bias") n d 1.0e-5)
        projected (nn/linear normalized (weight agent "scorer.1.weight")
                             (weight agent "scorer.1.bias") n d d)
        _ (modernbert/gelu-erf! projected projected (* n d))]
    (nn/linear projected (weight agent "scorer.3.weight")
               (weight agent "scorer.3.bias") n d 1)))

(defn- softmax [xs]
  (let [mx (reduce max Double/NEGATIVE_INFINITY xs)
        es (mapv #(Math/exp (- (double %) mx)) xs)
        sum (reduce + es)]
    (mapv #(/ % sum) es)))

(defn- action-probability
  [agent ^floats hidden logits]
  (let [d (:d-model (:encoder agent))
        p (softmax logits)
        sorted (sort > p)
        top1 (double (first sorted))
        top2 (double (second sorted))
        k (count p)
        entropy (/ (- (reduce + (map #(if (pos? %) (* % (Math/log %)) 0.0) p)))
                   (Math/log (double (max 2 k))))
        input (float-array (+ d 4))]
    (System/arraycopy hidden 0 input 0 d)
    (doseq [[i x] (map-indexed vector [top1 (- top1 top2) entropy (/ k 255.0)])]
      (aset input (+ d i) (float x)))
    (let [h (nn/linear input (weight agent "act_head.0.weight")
                       (weight agent "act_head.0.bias") 1 (+ d 4) 256)
          _ (modernbert/gelu-erf! h h 256)
          z (nn/linear h (weight agent "act_head.2.weight")
                       (weight agent "act_head.2.bias") 1 256 2)]
      (first (softmax z)))))

(defn- temperature-bucket [type k]
  (str type ":" (cond (<= k 2) "2" (<= k 5) "3-5"
                       (<= k 10) "6-10" :else "11+")))

(defn- confidence [p]
  (let [k (count p)]
    (if (< k 2) 1.0
        (max 0.0 (min 1.0
                      (- 1.0 (/ (- (reduce + (map #(if (pos? %) (* % (Math/log %)) 0.0) p)))
                                (Math/log (double k)))))))))

(defn- round4 [x]
  (/ (double (Math/round (* 10000.0 (double x)))) 10000.0))

(defn- infer-question
  [agent state question]
  (let [type (name (getk question "type"))
        cfg (:config agent)
        built (build-sequence (:tokenizer agent) state question
                              {:max-len (get cfg "max_len" 512)
                               :head-max-len (get cfg "head_max_len" 192)})
        markers (:markers built)
        options (render-options question)
        _ (when-not (= (count markers) (count options))
            (throw (ex-info "question options exceed Laya head token budget"
                            {:type type :options (count options)
                             :markers (count markers)
                             :head-max-len (get cfg "head_max_len")})))
        hidden (decision-hidden agent (:ids built) (:question-type built))
        logits (vec (score-markers agent hidden markers))
        type-index (get question-types type)
        default-temp (nth (get cfg "temperature" [1.0 1.0 1.0]) type-index)
        scale (double (get (get cfg "temperature_by_options" {})
                           (temperature-bucket type (count markers)) default-temp))
        probabilities (softmax (mapv #(/ (double %) (max 1.0e-3 scale)) logits))
        conf (round4 (confidence probabilities))
        action {:act-probability (round4 (action-probability agent hidden logits))}]
    {:tokens (count (:ids built))
     :answer
     (case type
       "choice"
       (let [criteria (getk question "criteria")
             labels (if (map? criteria) (vec (keys criteria)) (vec criteria))
             winner (first (apply max-key second (map-indexed vector probabilities)))]
         {:type :choice
          :choice (nth labels winner)
          :probabilities (into (array-map) (map vector labels (map round4 probabilities)))
          :confidence conf :action action})
       "score"
       (let [criteria (vec (getk question "criteria"))]
         {:type :score
          :score (round4 (reduce + (map-indexed #(* %1 %2) probabilities)))
          :legend (into (array-map) (map-indexed #(vector (str %1) %2) criteria))
          :probabilities (into (array-map)
                               (map-indexed #(vector (str %1) (round4 %2)) probabilities))
          :confidence conf :action action})
       "noul"
       {:type :noul :noul (round4 (nth probabilities 1))
        :confidence (round4 (max (nth probabilities 1)
                                 (- 1.0 (nth probabilities 1))))
        :action action})}))

(defn- answer-from-logits
  [agent question ^floats first-row logits]
  (let [type (name (getk question "type"))
        cfg (:config agent)
        k (count logits)
        type-index (get question-types type)
        default-temp (nth (get cfg "temperature" [1.0 1.0 1.0]) type-index)
        scale (double (get (get cfg "temperature_by_options" {})
                           (temperature-bucket type k) default-temp))
        probabilities (softmax (mapv #(/ (double %) (max 1.0e-3 scale)) logits))
        conf (round4 (confidence probabilities))
        action {:act-probability (round4 (action-probability agent first-row logits))}]
    (case type
      "choice"
      (let [criteria (getk question "criteria")
            labels (if (map? criteria) (vec (keys criteria)) (vec criteria))
            winner (first (apply max-key second (map-indexed vector probabilities)))]
        {:type :choice :choice (nth labels winner)
         :probabilities (into (array-map) (map vector labels (map round4 probabilities)))
         :confidence conf :action action})

      "score"
      (let [criteria (vec (score-criteria (getk question "criteria")))]
        {:type :score
         :score (round4 (reduce + (map-indexed #(* %1 %2) probabilities)))
         :legend (into (array-map) (map-indexed #(vector (str %1) %2) criteria))
         :probabilities (into (array-map)
                              (map-indexed #(vector (str %1) (round4 %2)) probabilities))
         :confidence conf :action action})

      "noul"
      {:type :noul :noul (round4 (nth probabilities 1))
       :confidence (round4 (max (nth probabilities 1)
                                (- 1.0 (nth probabilities 1))))
       :action action})))

(defn predict
  "Evaluate every typed question in one padded encoder/head pass. Matrix
  operations are batched while attention is segmented by each valid prefix."
  [agent state questions]
  (when (empty? questions)
    (throw (ex-info "Laya requires at least one question" {})))
  (doseq [[id question] questions]
    (when-not (map? question)
      (throw (ex-info "Laya question must be a map" {:id id :question question})))
    (let [type (getk question "type")
          type (if (keyword? type) (name type) type)]
      (when-not (contains? question-types type)
        (throw (ex-info "unknown Laya question type"
                        {:id id :type type})))
      (when-not (hask? question "instructions")
        (throw (ex-info "Laya question requires instructions" {:id id})))
      (when (#{"choice" "score"} type)
        (let [criteria (getk question "criteria")
              valid-criteria? (or (map? criteria) (sequential? criteria))
              options (when valid-criteria?
                        (count criteria))]
          (when (or (nil? options) (< options 2))
            (throw (ex-info "Laya choice and score questions require at least two options"
                            {:id id :type type :options options})))))))
  (let [cfg (:config agent)
        entries (vec questions)
        items (mapv (fn [[id question]]
                      (let [built (build-sequence
                                   (:tokenizer agent) state question
                                   {:max-len (get cfg "max_len" 512)
                                    :head-max-len (get cfg "head_max_len" 192)})
                            options (render-options question)]
                        (when-not (= (count (:markers built)) (count options))
                          (throw (ex-info "question options exceed Laya head token budget"
                                          {:id id :options (count options)
                                           :markers (count (:markers built))
                                           :head-max-len (get cfg "head_max_len")})))
                        (assoc built :id id :question question)))
                    entries)
        {:keys [data lengths max-len]} (decision-hidden-batch agent items)
        d (:d-model (:encoder agent))
        answers
        (into (array-map)
              (map-indexed
               (fn [b {:keys [id question markers]}]
                 (let [len (long (nth lengths b))
                       segment (copy-rows data (* b max-len) len d)
                       logits (vec (score-markers agent segment markers))
                       first-row (copy-rows segment 0 1 d)]
                   [id (answer-from-logits agent question first-row logits)]))
               items))]
    {:model :laya
     :answers answers
     :usage {:input-tokens (reduce + lengths) :output-tokens 0}}))
