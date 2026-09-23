(ns pretrained.decision
  "Typed-decision model API. Laya execution is added behind this task-level
  boundary; prompt construction remains independently testable."
  (:require [pretrained.decision.laya :as laya]))

(def registry laya/checkpoints)

(defn registered [] (sort (keys registry)))

(defn ensure-checkpoint
  "Download only the selected checkpoint from the bundled Laya repository."
  [key]
  (let [{:keys [hf] :as entry} (get registry key)]
    (when-not entry
      (throw (ex-info (str "unknown decision model " key)
                      {:model key :registered (registered)})))
    ((requiring-resolve 'pretrained.hub/ensure-model)
     hf (select-keys entry [:include :snapshot :subfolder]))))

(defn load-decision
  "Load a callable typed-decision checkpoint, downloading only that checkpoint
  when no explicit local directory is supplied. Invoke the result directly as
  `(model state questions)` or pass it to `predict`. Pass `{:gpu? true}` as
  the second argument for a lazy resident GPU agent; it is Closeable."
  ([key] (load-decision key nil {}))
  ([key dir-or-opts]
   (if (map? dir-or-opts)
     (load-decision key (:dir dir-or-opts) dir-or-opts)
     (load-decision key dir-or-opts {})))
  ([key dir opts]
   (let [dir (or dir (ensure-checkpoint key))
         cpu (assoc (laya/load-agent dir) ::entry (get registry key))]
     (if (:gpu? opts)
       ((requiring-resolve 'pretrained.decision.laya-gpu/gpu-agent) cpu opts)
       cpu))))

(defn predict
  "Run a loaded decision model in process through the backend-neutral callable
  contract `(agent state questions)`. LayaAgent implements it today; another
  decision backend can be attached without this facade depending on its code."
  [agent state questions]
  (agent state questions))
