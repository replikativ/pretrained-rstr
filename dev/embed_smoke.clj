(ns embed-smoke
  "Grid S4 end-to-end smoke: texts → pretrained.embed → proximum HNSW search →
  umap.rstr/fit. The einbetten pattern (eid-in-metadata, :cosine) without Python.
  Run: (load-file \"dev/embed_smoke.clj\") on a REPL with the :dev alias."
  (:require [pretrained.embed :as emb]
            [proximum.core :as prox]
            [umap :as umap]))

(defn run-smoke [model]
  (let [docs ["The Eiffel Tower stands in Paris, France."
              "Paris is the capital and largest city of France."
              "Datahike is a durable Datalog database for Clojure."
              "Datalog queries express joins declaratively over triples."
              "Gravitational waves are ripples in spacetime curvature."
              "LIGO detected gravitational waves from merging black holes."]
        E (emb/embed-texts model docs)
        vs (emb/rows E)
        ;; proximum: einbetten-style — eid in metadata, cosine distance
        base (str "/tmp/embed-smoke-" (System/currentTimeMillis))
        _ (.mkdirs (java.io.File. (str base "/mmap")))
        idx (reduce (fn [ix [i v]] (prox/insert ix v (long i) {:entity-id i}))
                    (prox/create-index {:type :hnsw :dim (:dim E) :distance :cosine
                                        :capacity 64
                                        :store-config {:backend :file :path (str base "/store")
                                                       :id (random-uuid)}
                                        :mmap-dir (str base "/mmap")})
                    (map-indexed vector vs))
        q (first (emb/rows (emb/embed-texts model "Which city is the capital of France?"
                                            :instruct "Given a question, retrieve the passage that answers it")))
        hits (prox/search idx q 3)
        ;; umap: flat double[] input
        u (umap/fit (emb/flat-doubles E) (:n E) (:dim E) :k 3 :out-dim 2 :n-epochs 50)]
    (println "query top-3 docs:" (mapv (fn [{:keys [id distance]}]
                                         [(subs (nth docs id) 0 30) (format "%.3f" distance)])
                                       hits))
    (println "umap emb shape:" (:n u) "x" (:dim u) "first point:"
             [(aget ^doubles (:emb u) 0) (aget ^doubles (:emb u) 1)])
    {:hits hits :umap u}))
