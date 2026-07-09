(ns pretrained.hub
  "Minimal HuggingFace Hub downloader — auto-fetch model repos into a flat local
  cache (fastembed/llama.cpp-style), no Python deps.

    (ensure-model \"Qwen/Qwen3-ASR-0.6B-hf\")  ;; → cache dir path, downloads if needed

  Design (verified against live endpoints, see .internal/hf_download_design.md):
  revision PINNED to the commit sha from the API (no torn snapshots), downloads via
  /resolve/{sha}/{file} with follow-redirects (Xet repos bridge transparently),
  resume via Range on .part files, sha256 verified while streaming for LFS files,
  atomic rename. HF_TOKEN honored (gated repos); note anonymous 401 also means
  'repo not found'. Cache: ~/.cache/raster/models/{org}--{name}/ + .files.json."
  (:require [clojure.data.json :as json]
            [clojure.java.io :as io]
            [clojure.string :as str])
  (:import [java.net URI]
           [java.net.http HttpClient HttpClient$Redirect HttpRequest
            HttpRequest$Builder HttpResponse$BodyHandlers]
           [java.nio.file Files Paths StandardCopyOption]
           [java.security MessageDigest DigestInputStream]))

(def ^:dynamic *cache-dir*
  (or (System/getenv "RASTER_MODELS_CACHE")
      (str (System/getProperty "user.home") "/.cache/raster/models")))

(def ^:private allow-re
  #"(?i)(\.safetensors|\.json|\.jinja|tokenizer\.model|merges\.txt|vocab\..*)$")

(def ^:private ^HttpClient client
  (-> (HttpClient/newBuilder) (.followRedirects HttpClient$Redirect/NORMAL) (.build)))

(defn- req ^HttpRequest [url]
  (let [b (-> (HttpRequest/newBuilder (URI/create url))
              (.header "User-Agent" "pretrained-rstr/0.1; jvm")
              (.header "Accept-Encoding" "identity"))
        b (if-let [tok (System/getenv "HF_TOKEN")]
            (.header ^HttpRequest$Builder b "Authorization" (str "Bearer " tok)) b)]
    (.build b)))

(defn get-json [url]
  (let [resp (.send client (req url) (HttpResponse$BodyHandlers/ofString))]
    (case (.statusCode resp)
      200 (json/read-str (.body resp))
      (401 403) (throw (ex-info (str "HF " (.statusCode resp) " for " url
                                     " — repo missing, private, or gated; set HF_TOKEN?") {}))
      (throw (ex-info (str "HF error " (.statusCode resp) " for " url) {})))))

(defn- sha256-hex [^MessageDigest md]
  (apply str (map #(format "%02x" %) (.digest md))))

(defn download-file!
  "Stream one file to dir, resume-aware, sha256-checked when expected. Returns path."
  [repo sha {:strs [path size] :as entry} dir]
  (let [dest (io/file dir path)
        lfs-sha (get-in entry ["lfs" "oid"])]
    (if (and (.exists dest) (= (.length dest) (long size)))
      dest
      (let [part (io/file dir (str path ".part"))
            _ (io/make-parents part)
            resume (if (.exists part) (.length part) 0)
            url (str "https://huggingface.co/" repo "/resolve/" sha "/" path)
            r (let [b (-> (HttpRequest/newBuilder (URI/create url))
                          (.header "User-Agent" "pretrained-rstr/0.1; jvm")
                          (.header "Accept-Encoding" "identity"))
                    b (if-let [tok (System/getenv "HF_TOKEN")]
                        (.header b "Authorization" (str "Bearer " tok)) b)
                    b (if (pos? resume) (.header b "Range" (str "bytes=" resume "-")) b)]
                (.build b))
            resp (.send client r (HttpResponse$BodyHandlers/ofInputStream))
            code (.statusCode resp)]
        (when (and (pos? resume) (not= code 206))          ;; server ignored Range
          (.delete part))
        (when-not (#{200 206} code)
          (throw (ex-info (str "HF " code " downloading " path
                               (when (#{401 403} code) " — gated/missing; set HF_TOKEN?")) {})))
        (let [fresh? (or (zero? resume) (not= code 206))
              md (when (and lfs-sha fresh?) (MessageDigest/getInstance "SHA-256"))
              in (if md (DigestInputStream. (.body resp) md) (.body resp))]
          (with-open [in in
                      out (java.io.FileOutputStream. part (boolean (not fresh?)))]
            (io/copy in out :buffer-size 262144))
          (when-not (= (.length part) (long size))
            (throw (ex-info (str path ": size mismatch " (.length part) " != " size
                             " (partial download — rerun to resume)") {})))
          (when (and md (not= (sha256-hex md) lfs-sha))
            (.delete part)
            (throw (ex-info (str path ": sha256 mismatch — corrupted download, deleted") {})))
          (Files/move (.toPath part) (.toPath dest)
                      (into-array java.nio.file.CopyOption [StandardCopyOption/ATOMIC_MOVE]))
          dest)))))

(defn ensure-model
  "Ensure `repo` (e.g. \"Qwen/Qwen3-ASR-0.6B-hf\") is in the local cache; download
  what's missing. Returns the model dir path. Offline-tolerant: if the API is
  unreachable but a complete cached copy exists, uses it."
  [repo]
  (let [dir (io/file *cache-dir* (str/replace repo "/" "--"))
        manifest (io/file dir ".files.json")
        complete? (fn []
                    (and (.exists manifest)
                         (every? (fn [{:strs [path size]}]
                                   (let [f (io/file dir path)]
                                     (and (.exists f) (= (.length f) (long size)))))
                                 (json/read-str (slurp manifest)))))]
    (if (complete?)
      (.getPath dir)
      (let [info (try (get-json (str "https://huggingface.co/api/models/" repo))
                      (catch Exception e
                        (if (complete?) nil (throw e))))]
        (if (nil? info)
          (.getPath dir)
          (let [sha (get info "sha")
                tree (get-json (str "https://huggingface.co/api/models/" repo
                                    "/tree/main?recursive=true"))
                files (filterv #(re-find allow-re (get % "path")) tree)]
            (.mkdirs dir)
            (doseq [f files]
              (loop [attempt 1]
                (let [r (try (download-file! repo sha f (str dir)) ::ok
                             (catch Exception e (if (>= attempt 3) (throw e) ::retry)))]
                  (when (= r ::retry)
                    (Thread/sleep (* 2000 attempt))
                    (recur (inc attempt))))))
            (spit manifest (json/write-str (mapv #(select-keys % ["path" "size"]) files)))
            (spit (io/file dir ".revision") sha)
            (.getPath dir)))))))
