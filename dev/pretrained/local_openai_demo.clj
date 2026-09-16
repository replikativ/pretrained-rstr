(ns pretrained.local-openai-demo
  "Resource-gated single-process OpenAI smoke over a real local model.

  Opens `pretrained.openai.local/open-server` on an ephemeral port, runs a
  two-turn conversation through the chat endpoint, and checks that the second
  turn reports resident cached tokens. Run with:

      clojure -M:valhalla:openai-server:real-cluster-test -e \\
        \"(require '[pretrained.local-openai-demo :as demo]) (demo/run!)\""
  (:refer-clojure :exclude [run!])
  (:require [clojure.data.json :as json]
            [org.httpkit.client :as http-client]
            [pretrained.host-resources :as host-resources]
            [pretrained.openai.local :as local-server])
  (:import (java.io Closeable)))

(defn- chat!
  [port model-id messages max-new-tokens]
  (let [response
        @(http-client/post
          (str "http://127.0.0.1:" port "/v1/chat/completions")
          {:timeout 120000
           :headers {"content-type" "application/json"}
           :body (json/write-str {:model model-id
                                  :messages messages
                                  :max_completion_tokens max-new-tokens})})]
    (when-not (= 200 (:status response))
      (throw (ex-info "OpenAI completion failed" {:response response})))
    (json/read-str (:body response) :key-fn keyword)))

(defn run!
  "Serve `~/Development/models/gemma-3-270m-it` in one process and prove reuse.

  Options: `:model-directory`, `:model-id`, `:max-new-tokens` (default 16),
  `:resource-thresholds`, `:force?`, and any `pretrained.openai.local`
  server option under `:server`."
  ([] (run! {}))
  ([opts]
   (let [admission (host-resources/preflight (:resource-thresholds opts))]
     (when (and (not (:force? opts)) (not (:admitted? admission)))
       (throw (ex-info "Refusing real-model compile under current host pressure"
                       admission)))
     (let [model-directory (or (:model-directory opts)
                               (str (System/getProperty "user.home")
                                    "/Development/models/gemma-3-270m-it"))
           model-id (:model-id opts "gemma-3-270m-it")
           max-new (:max-new-tokens opts 16)
           server (local-server/open-server
                   (merge {:model-directory model-directory
                           :model-id model-id
                           :port 0}
                          (:server opts)))]
       (try
         (let [port (local-server/local-port server)
               first-messages [{:role "user"
                                :content "Name the capital of France in one word."}]
               first-response (chat! port model-id first-messages max-new)
               first-text (get-in first-response [:choices 0 :message :content])
               second-messages (conj first-messages
                                     {:role "assistant" :content first-text}
                                     {:role "user"
                                      :content "And the capital of Germany?"})
               second-response (chat! port model-id second-messages max-new)
               cached (get-in second-response
                              [:usage :prompt_tokens_details :cached_tokens])]
           (when-not (pos? (long cached))
             (throw (ex-info "Second turn did not reuse resident KV"
                             {:first-response first-response
                              :second-response second-response})))
           {:resource-admission admission
            :model model-id
            :first-text first-text
            :second-text (get-in second-response [:choices 0 :message :content])
            :second-usage (:usage second-response)
            :cached-tokens cached})
         (finally
           (.close ^Closeable server)))))))
