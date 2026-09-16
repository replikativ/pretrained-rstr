(ns pretrained.chat
  "Chat-template rendering and stop-token discovery for supported model families.

  Hugging Face checkpoints carry a Jinja `chat_template`. This namespace does
  not interpret Jinja; it recognizes the two template families used by the
  curated decoder registry and renders them directly:

  - Gemma: `<start_of_turn>role\\n...<end_of_turn>\\n`, assistant rendered as
    `model`, a system message folded into the first user turn. The tokenizer
    prepends `<bos>`, so the rendered text does not.
  - ChatML (Qwen, SmolLM): `<|im_start|>role\\n...<|im_end|>\\n`.

  Any other template falls back to joining message contents with newlines,
  which is what an unrecognized instruct model receives today."
  (:require [clojure.data.json :as json]
            [clojure.java.io :as io]
            [clojure.string :as str]))

(defn- read-json
  [file]
  (when (.exists (io/file file))
    (json/read-str (slurp file))))

(defn template-family
  "Return `:gemma`, `:chatml`, or nil for the chat template in `dir`."
  [dir]
  (let [template (get (read-json (str dir "/tokenizer_config.json"))
                      "chat_template")]
    (when (string? template)
      (cond
        (str/includes? template "<start_of_turn>") :gemma
        (str/includes? template "<|im_start|>") :chatml
        :else nil))))

(defn- content-string
  [message]
  (let [content (:content message)]
    (cond
      (string? content) content
      (sequential? content) (str/join (map #(get % :text (get % "text" ""))
                                           content))
      :else (str content))))

(defn- render-gemma
  [messages]
  (let [[first-message & _] messages
        system? (= "system" (:role first-message))
        prefix (if system? (str (content-string first-message) "\n\n") "")
        turns (if system? (rest messages) messages)]
    (str
     (apply str
            (map-indexed
             (fn [index message]
               (let [role (if (= "assistant" (:role message))
                            "model"
                            (:role message))
                     content (str (when (zero? index) prefix)
                                  (content-string message))]
                 (str "<start_of_turn>" role "\n" content "<end_of_turn>\n")))
             turns))
     "<start_of_turn>model\n")))

(defn- render-chatml
  [messages]
  (str
   (apply str
          (map (fn [message]
                 (str "<|im_start|>" (:role message) "\n"
                      (content-string message) "<|im_end|>\n"))
               messages))
   "<|im_start|>assistant\n"))

(defn- render-plain
  [messages]
  (str/join "\n" (map content-string messages)))

(defn render-fn
  "Return a function from normalized OpenAI messages to prompt text for `dir`.

  Messages are maps with string `:role` and `:content`, as produced by
  `pretrained.openai/normalize-chat-request`. Roles `developer` are rendered
  as `system`."
  [dir]
  (let [render (case (template-family dir)
                 :gemma render-gemma
                 :chatml render-chatml
                 render-plain)]
    (fn [messages]
      (render (mapv (fn [message]
                      (update message :role
                              #(if (= "developer" %) "system" %)))
                    messages)))))

(defn eos-ids
  "Return the set of token ids that end generation for the checkpoint in `dir`.

  Combines `generation_config.json` `eos_token_id` (an integer or a list) with
  the id of `tokenizer_config.json`'s `eos_token`, resolved through
  `tokenizer.json`'s added tokens and vocabulary. Returns an empty set when
  none can be found."
  [dir]
  (let [generation (read-json (str dir "/generation_config.json"))
        configured (get generation "eos_token_id")
        configured (cond
                     (integer? configured) [configured]
                     (sequential? configured) configured
                     :else [])
        eos-token (get (read-json (str dir "/tokenizer_config.json"))
                       "eos_token")
        eos-token (if (map? eos-token) (get eos-token "content") eos-token)
        tokenizer (when (string? eos-token)
                    (read-json (str dir "/tokenizer.json")))
        added (into {}
                    (map (fn [token] [(get token "content") (get token "id")]))
                    (get tokenizer "added_tokens"))
        vocab (get-in tokenizer ["model" "vocab"])
        named (when (string? eos-token)
                (or (get added eos-token) (get vocab eos-token)))]
    (into #{} (comp (remove nil?) (map long))
          (cond-> (vec configured) named (conj named)))))
