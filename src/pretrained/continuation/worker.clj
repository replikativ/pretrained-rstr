(ns pretrained.continuation.worker
  "Worker-local execution loop for resident paged decode continuations.

  This namespace joins the pure scheduler to the stateful paged decoder. Cache
  restore/admission happens before a request enters this ready queue; checkpoint,
  retention, and eviction policy consume the completed/retired outputs after an
  iteration. The boundary keeps Datahike/Konserve control-plane I/O off the GPU
  submission path."
  (:require [pretrained.continuation.paged-decoder :as paged-decoder]
            [pretrained.continuation.scheduler :as scheduler]))

(defn run-decode-iteration!
  "Plan and execute one worker-local continuous decode iteration.

  `decoder` is a fixed-capacity paged decoder. `previous-lanes` is the lane table
  returned by the preceding call, and `requests` contains every currently ready
  decode request, including retained occupants plus new arrivals. Requests must
  already have resident routes and the scheduler fields required by
  `decode-submission`.

  Options accept `:eos-ids`. The result includes the next `:lanes`, runnable and
  completed requests, deferred arrivals, lane results, and both current- and
  next-iteration protected continuation sets. No GPU work is submitted when the
  ready set is empty."
  [decoder previous-lanes requests & {:keys [eos-ids] :or {eos-ids #{}}}]
  (let [capacity (long (get-in decoder [:decode-state :batch-size] 1))
        plan (scheduler/plan-decode-lanes capacity previous-lanes requests)
        submission (scheduler/decode-submission plan)
        lane-work (:lane-work submission)
        results
        (if (seq lane-work)
          (do
            (when (seq (:prime-lanes submission))
              (paged-decoder/prime-lanes! decoder (:prime-lanes submission)))
            (paged-decoder/step-lanes! decoder lane-work))
          [])
        completed (scheduler/complete-decode-iteration
                   plan results {:eos-ids eos-ids})
        next-protected
        (into #{}
              (keep #(or (:request/continuation-id %) (:request/id %)))
              (:lanes completed))]
    {:lanes (:lanes completed)
     :runnable (:runnable completed)
     :completed (:completed completed)
     :deferred (:deferred plan)
     :results results
     :retained (:retained plan)
     :refilled (:refill plan)
     :retired (:retired plan)
     :protected-continuation-ids (:protected-continuation-ids submission)
     :next-protected-continuation-ids next-protected}))

(defn next-ready-requests
  "Combine retained runnable work, deferred ready work, and new arrivals.

  This helper preserves the explicit iteration boundary: callers may first
  checkpoint completions, restore new routes, or reject arrivals before passing
  the resulting vector to the next `run-decode-iteration!` call."
  [iteration new-arrivals]
  (vec (concat (:runnable iteration) (:deferred iteration) new-arrivals)))

(defn generate-continuously!
  "Generate independently retiring sequences through the worker iteration loop.

  `continuation-ids` and `prompts` must fill the decoder's fixed batch and meet
  `paged-decoder/prime-prompts-batch!`'s equal-work suffix contract. Each lane
  may then retire independently on `eos-ids` or after `max-new` tokens while the
  remaining lanes continue through sparse submissions. Returns one generated
  token vector per input lane, including a terminating EOS token."
  [decoder continuation-ids prompts max-new & {:keys [eos-ids] :or {eos-ids #{}}}]
  (when-not (and (integer? max-new) (not (neg? max-new)))
    (throw (ex-info "Maximum generated tokens must be a non-negative integer"
                    {:max-new max-new})))
  (let [continuation-ids (vec continuation-ids)
        prompts (mapv vec prompts)
        capacity (long (get-in decoder [:decode-state :batch-size] 1))]
    (when-not (and (= capacity (count continuation-ids))
                   (= capacity (count prompts)))
      (throw (ex-info "Continuous generation must fill the fixed decode batch"
                      {:batch-size capacity
                       :continuation-count (count continuation-ids)
                       :prompt-count (count prompts)})))
    (paged-decoder/prime-prompts-batch! decoder continuation-ids prompts)
    (if (zero? max-new)
      (vec (repeat capacity []))
      (let [initial
            (mapv (fn [lane continuation-id prompt]
                    #:request{:id continuation-id
                              :continuation-id continuation-id
                              :phase :decode
                              :remaining-tokens max-new
                              :pending-token (peek prompt)
                              :position (dec (count prompt))
                              :arrival lane})
                  (range capacity) continuation-ids prompts)
            output-index (zipmap continuation-ids (range capacity))]
        (loop [lanes initial
               ready initial
               output (vec (repeat capacity []))]
          (if (seq ready)
            (let [iteration (run-decode-iteration!
                             decoder lanes ready :eos-ids eos-ids)
                  output
                  (reduce (fn [rows {:keys [continuation-id token]}]
                            (update rows (output-index continuation-id) conj token))
                          output (:results iteration))]
              (recur (:lanes iteration)
                     (next-ready-requests iteration [])
                     output))
            output))))))
