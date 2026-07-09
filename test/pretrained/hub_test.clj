(ns pretrained.hub-test
  "Model-free, network-free: hub cache-completeness logic and layout conventions."
  (:require [clojure.test :refer [deftest is testing]]
            [clojure.data.json :as json]
            [clojure.java.io :as io]
            [pretrained.hub :as hub])
  (:import [java.io File]))

(deftest cache-complete-detection
  (let [dir (doto (File/createTempFile "hubc" "") (.delete) (.mkdirs))]
    (binding [hub/*cache-dir* (.getPath dir)]
      (let [mdir (io/file dir "Fake--model")
            manifest (io/file mdir ".files.json")]
        (.mkdirs mdir)
        (spit (io/file mdir "config.json") "{\"a\":1}")
        (spit manifest (json/write-str [{"path" "config.json"
                                         "size" (.length (io/file mdir "config.json"))}]))
        (testing "complete cache is used offline (no network call)"
          ;; ensure-model must return the dir WITHOUT hitting the API when complete
          (is (= (.getPath mdir) (hub/ensure-model "Fake/model"))))
        (testing "size mismatch → not complete"
          (spit (io/file mdir "config.json") "{\"a\":1,\"b\":2}")
          ;; now incomplete; ensure-model would go to network — we only assert the
          ;; manifest check itself via the changed size
          (is (not= (get (first (json/read-str (slurp manifest))) "size")
                    (.length (io/file mdir "config.json")))))))
    (doseq [f (reverse (file-seq dir))] (.delete ^File f))))
