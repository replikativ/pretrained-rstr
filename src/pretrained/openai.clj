(ns pretrained.openai
  "Pure OpenAI-compatible chat-completion boundary.

  This namespace owns JSON-shaped request and response values, not HTTP, GPU
  work, or cluster routing. A server supplies model resolution, chat templating
  and tokenization, then submits the returned transport-neutral generation
  request to the continuation controller. Controller deliveries are converted
  to streamed chunks or one terminal response with the same immutable context."
  (:require [clojure.string :as str]
            [clojure.data.json :as json]
            [pretrained.continuation.controller.protocol :as protocol]))

(defn- field
  [m k]
  (or (get m k) (get m (name k))))

(defn- invalid!
  [message param value]
  (throw (ex-info message
                  {:openai/error {:message message
                                  :type "invalid_request_error"
                                  :param (name param)
                                  :code nil}
                   :param param
                   :value value})))

(defn- normalize-message
  [message]
  (let [role (field message :role)
        content (field message :content)]
    (when-not (contains? #{"system" "developer" "user" "assistant"} role)
      (invalid! "Only text system, developer, user, and assistant messages are supported"
                :messages message))
    (when-not (string? content)
      (invalid! "Message content must be a string" :messages message))
    {:role role :content content}))

(defn- positive-token-limit
  [body default-limit]
  (let [limit (or (field body :max_completion_tokens)
                  (field body :max_tokens)
                  default-limit)]
    (when-not (and (integer? limit) (pos? limit))
      (invalid! "max_completion_tokens must be a positive integer"
                :max_completion_tokens limit))
    (long limit)))

(defn- sampling
  [body]
  (let [temperature (or (field body :temperature) 1.0)
        top-p (or (field body :top_p) 1.0)
        stop (field body :stop)
        stop (cond
               (nil? stop) []
               (string? stop) [stop]
               (and (sequential? stop) (every? string? stop)) (vec stop)
               :else (invalid! "stop must be a string or an array of strings"
                               :stop stop))]
    (when-not (and (number? temperature) (<= 0.0 (double temperature) 2.0))
      (invalid! "temperature must be between 0 and 2" :temperature temperature))
    (when-not (and (number? top-p) (< 0.0 (double top-p))
                   (<= (double top-p) 1.0))
      (invalid! "top_p must be greater than 0 and at most 1" :top_p top-p))
    {:temperature (double temperature)
     :top-p (double top-p)
     :stop stop
     :seed (field body :seed)}))

(defn normalize-chat-request
  "Normalize the supported `POST /v1/chat/completions` request subset.

  Options require `:models`, a map from public model id to model fingerprint,
  and `:tokenize-chat`, a function from normalized messages to a nonempty token
  collection. Optional deterministic hooks are `:request-id`, `:created`, and
  `:default-max-new-tokens` (default 256).

  The result contains an OpenAI response context and a validated continuation
  generation request. Unsupported fields may be present and are ignored; known
  fields with unsupported values fail as OpenAI-shaped `ExceptionInfo`."
  [body {:keys [models tokenize-chat request-id created default-max-new-tokens]
         :or {request-id #(str "chatcmpl-" (random-uuid))
              created #(quot (System/currentTimeMillis) 1000)
              default-max-new-tokens 256}}]
  (when-not (map? body)
    (invalid! "Request body must be a JSON object" :body body))
  (when-not (ifn? tokenize-chat)
    (throw (ex-info "OpenAI adapter requires :tokenize-chat" {})))
  (let [model (field body :model)
        fingerprint (get models model)
        raw-messages (field body :messages)]
    (when-not (and (string? model) (not (str/blank? model)))
      (invalid! "model is required" :model model))
    (when-not (string? fingerprint)
      (invalid! "The requested model is not available" :model model))
    (when-not (and (sequential? raw-messages) (seq raw-messages))
      (invalid! "messages must be a nonempty array" :messages raw-messages))
    (let [messages (mapv normalize-message raw-messages)
          tokens (vec (tokenize-chat messages))
          id (request-id)
          max-new (positive-token-limit body default-max-new-tokens)
          sampling (sampling body)]
      (when-not (and (seq tokens) (every? integer? tokens))
        (throw (ex-info "Chat tokenizer must return nonempty integer tokens"
                        {:model model :tokens tokens})))
      {:openai/context
       {:id id
        :object "chat.completion"
        :created (long (created))
        :model model
        :stream? (true? (field body :stream))
        :prompt-tokens (count tokens)
        :request/max-new-tokens max-new
        :include-usage? (true? (field (field body :stream_options)
                                      :include_usage))}
       :generation/request
       (protocol/generation-request
        {:request/id id
         :request/model-fingerprint fingerprint
         :request/tokens tokens
         :request/max-new-tokens max-new
         :request/sampling sampling})})))

(defn error-object
  "Return an OpenAI-shaped error body from an exception or error value."
  [error]
  {:error
   (or (when (instance? clojure.lang.ExceptionInfo error)
         (:openai/error (ex-data error)))
       {:message (cond
                   (instance? Throwable error)
                   (or (.getMessage ^Throwable error) "Generation failed")

                   (keyword? error) (name error)
                   (string? error) error
                   (map? error) (or (:message error) (pr-str error))
                   :else (str (or error :generation-failed)))
        :type "server_error"
        :param nil
        :code nil})})

(defn- finish-reason
  [context result]
  (cond
    (= :length (:stop-reason result)) "length"
    (= :capacity (:stop-reason result)) "length"
    (= (count (:tokens result))
       (:request/max-new-tokens context)) "length"
    :else "stop"))

(defn stream-values
  "Convert one fenced controller delivery into zero or more SSE payload values.

  `context` is `:openai/context` augmented with the generation token limit and
  `:decode-token`, a function from one token id to its incremental text. Values
  are Clojure data except for the terminal sentinel `:done`; `sse` serializes
  them for an HTTP transport."
  [context delivery]
  (case (:response/type delivery)
    :delta
    [{:id (:id context)
      :object "chat.completion.chunk"
      :created (:created context)
      :model (:model context)
      :choices [{:index 0
                 :delta {:content ((:decode-token context)
                                   (:response/token delivery))}
                 :finish_reason nil}]}]

    :completed
    (let [result (:response/value delivery)
          completion-tokens (count (:tokens result))
          final-chunk
          {:id (:id context)
           :object "chat.completion.chunk"
           :created (:created context)
           :model (:model context)
           :choices [{:index 0 :delta {}
                      :finish_reason (finish-reason context result)}]}
          usage-chunk
          {:id (:id context)
           :object "chat.completion.chunk"
           :created (:created context)
           :model (:model context)
           :choices []
           :usage {:prompt_tokens (:prompt-tokens context)
                   :completion_tokens completion-tokens
                   :total_tokens (+ (:prompt-tokens context)
                                    completion-tokens)}}]
      (cond-> [final-chunk]
        (:include-usage? context) (conj usage-chunk)
        true (conj :done)))

    :cancelled
    [(error-object :cancelled) :done]

    :error
    [(error-object (:response/error delivery)) :done]

    []))

(defn completion-response
  "Build a non-streamed chat completion from a terminal controller delivery."
  [context delivery]
  (if (= :completed (:response/type delivery))
    (let [result (:response/value delivery)
          tokens (:tokens result)
          text ((:decode-tokens context) tokens)
          completion-tokens (count tokens)]
      {:id (:id context)
       :object "chat.completion"
       :created (:created context)
       :model (:model context)
       :choices [{:index 0
                  :message {:role "assistant" :content text}
                  :finish_reason (finish-reason context result)}]
       :usage {:prompt_tokens (:prompt-tokens context)
               :completion_tokens completion-tokens
               :total_tokens (+ (:prompt-tokens context) completion-tokens)}})
    (error-object (or (:response/error delivery) (:response/type delivery)))))

(defn sse
  "Serialize one `stream-values` value as an SSE data frame."
  [value]
  (str "data: " (if (= :done value) "[DONE]" (json/write-str value)) "\n\n"))
