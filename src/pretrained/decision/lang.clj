(ns pretrained.decision.lang
  "Dependency-free script and best-effort Latin-language detection for Laya
  checkpoint routing. Keys in structured state are intentionally ignored."
  (:require [clojure.string :as str]))

(def ^:private script-ranges
  [[:greek [[0x0370 0x03ff] [0x1f00 0x1fff]]]
   [:cyrillic [[0x0400 0x052f] [0x2de0 0x2dff] [0xa640 0xa69f]]]
   [:armenian [[0x0530 0x058f]]]
   [:hebrew [[0x0590 0x05ff]]]
   [:arabic [[0x0600 0x06ff] [0x0750 0x077f] [0x08a0 0x08ff]
             [0xfb50 0xfdff] [0xfe70 0xfeff]]]
   [:devanagari [[0x0900 0x097f] [0xa8e0 0xa8ff]]]
   [:bengali [[0x0980 0x09ff]]]
   [:gurmukhi [[0x0a00 0x0a7f]]]
   [:gujarati [[0x0a80 0x0aff]]]
   [:oriya [[0x0b00 0x0b7f]]]
   [:tamil [[0x0b80 0x0bff]]]
   [:telugu [[0x0c00 0x0c7f]]]
   [:kannada [[0x0c80 0x0cff]]]
   [:malayalam [[0x0d00 0x0d7f]]]
   [:sinhala [[0x0d80 0x0dff]]]
   [:thai [[0x0e00 0x0e7f]]]
   [:lao [[0x0e80 0x0eff]]]
   [:tibetan [[0x0f00 0x0fff]]]
   [:myanmar [[0x1000 0x109f]]]
   [:georgian [[0x10a0 0x10ff]]]
   [:ethiopic [[0x1200 0x137f]]]
   [:khmer [[0x1780 0x17ff]]]
   [:hangul [[0x1100 0x11ff] [0x3130 0x318f] [0xac00 0xd7af]]]
   [:kana [[0x3040 0x309f] [0x30a0 0x30ff] [0x31f0 0x31ff]]]
   [:han [[0x3400 0x4dbf] [0x4e00 0x9fff] [0xf900 0xfaff]]]])

(def ^:private stopwords
  {:en #{"the" "and" "is" "are" "was" "were" "to" "of" "in" "for" "with"
         "that" "this" "it" "you" "have" "has" "not" "but" "on" "at" "be"
         "as" "from" "will" "can" "would" "there" "their" "what" "which"
         "please" "we" "i"}
   :fr #{"le" "la" "les" "des" "une" "est" "pour" "dans" "que" "qui" "avec"
         "sur" "pas" "plus" "nous" "vous" "être" "cette" "mais" "sont" "ont"
         "aux" "ce"}
   :de #{"der" "die" "das" "und" "ist" "ein" "eine" "den" "dem" "nicht" "mit"
         "für" "auf" "von" "zu" "sich" "auch" "werden" "wurde" "haben" "sind"
         "oder" "aber"}
   :es #{"el" "los" "las" "que" "por" "con" "para" "una" "es" "se" "del"
         "como" "pero" "son" "está" "este" "esta" "todo" "más" "muy" "hay" "sus"}
   :pt #{"os" "as" "que" "em" "um" "uma" "para" "com" "não" "é" "se" "do"
         "da" "dos" "das" "mas" "são" "está" "este" "esta" "muito" "pelo" "pela"}
   :it #{"il" "lo" "gli" "che" "di" "per" "con" "non" "è" "si" "del" "della"
         "sono" "questo" "questa" "anche" "come" "più" "nella" "alla"}
   :nl #{"het" "een" "van" "is" "op" "te" "dat" "niet" "met" "voor" "zijn"
         "aan" "door" "maar" "ook" "worden" "deze" "naar" "wordt"}})

(def ^:private non-english-diacritics
  (set "àâäãáåçéèêëíìîïñóòôöõøúùûüýÿßæœđłşţğıåäö"))

(defn- text-leaves [state depth]
  (cond
    (> depth 6) []
    (nil? state) []
    (string? state) [state]
    (map? state) (mapcat #(text-leaves % (inc depth)) (vals state))
    (sequential? state) (mapcat #(text-leaves % (inc depth)) state)
    :else []))

(defn state-text
  "Flatten string leaves of state, ignoring map keys and limiting work."
  ([state] (state-text state 4000))
  ([state max-chars]
   (let [text (str/join " " (text-leaves state 0))]
     (subs text 0 (min max-chars (count text))))))

(defn- latin-codepoint? [cp]
  (or (< cp 0x0250) (<= 0x1e00 cp 0x1eff)))

(defn- script-of [cp]
  (some (fn [[script ranges]]
          (when (some (fn [[lo hi]] (<= lo cp hi)) ranges) script))
        script-ranges))

(defn script-profile [text]
  (let [counts
        (reduce (fn [m ch]
                  (if (Character/isLetter ^char ch)
                    (let [cp (int ch)
                          script (if (latin-codepoint? cp) :latin (script-of cp))]
                      (if script (update m script (fnil inc 0)) m))
                    m))
                {} text)
        total (reduce + 0 (vals counts))]
    (if (zero? total) {}
        (into {} (map (fn [[k n]] [k (/ (double n) total)]) counts)))))

(defn detect-script [text]
  (let [profile (script-profile text)]
    (if (empty? profile) :unknown
        (first (apply max-key second profile)))))

(defn guess-latin-language [text]
  (let [words (map str/lower-case (re-seq #"[^\W\d_]+" text))]
    (when (>= (count words) 4)
      (let [scores (into {} (map (fn [[lang ws]]
                                   [lang (count (filter ws words))]) stopwords))
            en (get scores :en 0)
            [best-lang best] (apply max-key second (dissoc scores :en))
            lowered (str/lower-case text)
            diacritics (count (filter non-english-diacritics lowered))
            rate (/ (double diacritics) (max 1 (count lowered)))]
        (cond
          (and (zero? best) (< rate 0.02)) (when (pos? en) :en)
          (>= best (max 2 (+ en 2))) best-lang
          (and (>= rate 0.04) (>= best en)) best-lang
          (pos? en) :en
          :else nil)))))

(defn analyse [state]
  (let [text (state-text state)
        profile (script-profile text)
        script (if (empty? profile) :unknown (first (apply max-key second profile)))
        non-latin (if (empty? profile) 0.0 (- 1.0 (get profile :latin 0.0)))]
    (cond
      (= script :unknown)
      {:script :unknown :script-profile profile :language nil
       :english? true :non-latin-fraction 0.0}

      (not= script :latin)
      {:script script :script-profile profile :language nil
       :english? false :non-latin-fraction (/ (Math/round (* non-latin 10000.0)) 10000.0)}

      :else
      (let [language (guess-latin-language text)]
        {:script :latin :script-profile profile :language language
         :english? (or (nil? language) (= language :en))
         :non-latin-fraction (/ (Math/round (* non-latin 10000.0)) 10000.0)}))))
