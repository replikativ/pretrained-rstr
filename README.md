# pretrained-rstr

[![Clojars Project](https://img.shields.io/clojars/v/org.replikativ/pretrained-rstr.svg)](https://clojars.org/org.replikativ/pretrained-rstr)
[![CircleCI](https://circleci.com/gh/replikativ/pretrained-rstr.svg?style=shield)](https://circleci.com/gh/replikativ/pretrained-rstr)
[![Slack](https://img.shields.io/badge/slack-join_chat-brightgreen.svg)](https://clojurians.slack.com/archives/C09622F337D)

Run pretrained Hugging Face models natively on the JVM and treat their
attention state as durable, forkable numerical memory.

pretrained-rstr loads safetensors directly, quantizes linear weights, and runs
embeddings, speech recognition, and decoder LLMs through
[Raster](https://github.com/replikativ/raster). Decoder state can be split into
immutable prefix-hashed chunks, indexed by
[Datahike](https://datahike.io/), stored through
[Konserve](https://github.com/replikativ/konserve), restored into resident GPU
pages, and shared copy-on-write between continuations.

> **Experimental:** APIs and checkpoint formats may change before 1.0. The
> repository contains tested research software, not a managed inference service.

## What is here

- Direct JVM inference without Python or an ONNX runtime.
- Descriptor-driven decoder architectures with CPU Q4/Q8 and resident GPU
  execution.
- Exact CPU, contiguous-GPU, and paged-GPU continuation boundaries.
- Content-addressed KV chunks, longest-prefix lookup, asynchronous publication,
  tiered replicas, and mmap restoration.
- Fixed-capacity continuous decode lanes, prefix-page sharing, copy-on-write,
  and explainable admission/eviction decisions.
- Model-free tests and demos for the control/data plane; hardware-gated parity
  anchors for real models.

The current proof is LLM inference. The same control-plane/data-plane split is a
promising basis for larger numerical simulations, but this project does not yet
provide a PDE solver, weather model, MPI runtime, or direct inter-node GPU
transport. See [Numerical memory beyond LLM inference](doc/numerical-memory.md).

## Install

Use JDK 21 or newer and add a released library version to `deps.edn`:

```clojure
{:deps {org.replikativ/pretrained-rstr {:mvn/version "0.1.48"}}}
```

Raster is pinned in `deps.edn`. OpenBLAS is required for
floating-point GEMM paths. ffmpeg is optional for non-WAV audio. GPU execution
requires Level Zero on Intel or a compatible OpenCL ICD on Intel, NVIDIA, or AMD.

## Quick start

Every task-level API follows the same shape: load a curated registry entry or a
known local model directory, then call the task verb.

```clojure
(require '[pretrained.embed :as emb]
         '[pretrained.asr :as asr]
         '[pretrained.lm :as lm])

;; Registry entries download pinned Hugging Face files on first use.
(def embedder (emb/load-embedder :qwen3-embedding-0.6b))
(emb/embed-texts embedder ["Durable numerical memory for model inference."])
;; => {:data float-array, :n 1, :dim 1024}

(def speech-model (asr/load-asr :moonshine-streaming-medium))
(asr/transcribe speech-model "voice-note.wav")
;; => "..."

(def language-model (lm/load-lm :gemma-3-270m-it))
(lm/generate-text language-model "The capital of France is" 20)
;; => " Paris..."

;; Select resident GPU execution while loading a supported model.
;; CPU decode honors {:temperature :top-k :top-p :seed}; GPU decode is greedy.
(def gpu-model (lm/load-lm :gemma-3-270m-it {:gpu? true}))
```

### Typed decisions with Laya

The three checkpoints bundled by `convaiinnovations/laya` are available without
Python. Only the selected checkpoint is downloaded, and every question in one
`predict` call shares a padded encoder/head pass.

```clojure
(require '[pretrained.decision :as decision]
         '[pretrained.decision.router :as laya-router])

(def agent (decision/load-decision :laya-english))

(def state
  {:from "user@acme.com"
   :subject "Duplicate charge"
   :body "Please refund the duplicate or we will cancel."})

(def questions
  {:department
   {:type :choice
    :instructions "Which department should handle this request?"
    :criteria {:billing "invoices, payments, refunds"
               :technical "bugs, outages, system errors"
               :other "everything else"}}
   :urgency
   {:type :score
    :instructions "How urgent is this request?"
    :criteria ["not urgent" "soon" "critical"]}
   :refund-requested
   {:type :noul
    :instructions "Does the user explicitly request a refund?"}})

(agent state questions)
;; => {:model :laya, :answers {...},
;;     :usage {:input-tokens ... :output-tokens 0}}

;; The named form is equivalent when it reads better at a call site.
(decision/predict agent state questions)

;; Route English to ModernBERT and non-English/scripted text to mmBERT.
(def route-decision
  (laya-router/router {:max-loaded 2
                       :preload [:english :multilingual]}))
(route-decision state questions)
;; Adds :routing with :model, :checkpoint, :reason, and detection metadata.

;; Route overrides use the callable three-argument form.
(route-decision state questions {:model :typed-decisions})

;; Routers are Closeable; this unloads every retained checkpoint.
(.close ^java.io.Closeable route-decision)
```

Registry keys are `:laya-english`, `:laya-multilingual`, and
`:laya-typed-decisions`. The router also supports explicit `:model`, `:task`,
and `:lang` overrides, opt-in typed-workflow detection, preloading, attachment
of an existing agent, unloading, and bounded LRU residency. For example:

```clojure
(laya-router/preload! router [:english :multilingual])
(laya-router/predict router hindi-state questions {:lang :hi})
(laya-router/unload! router)
```

Like Python's `agent.predict(state, questions)`, both forms execute entirely in
the current process. No Python runtime, subprocess, HTTP service, or JSON
round-trip is involved; Clojure maps and keywords go in and an immutable result
map comes back. The router is lazy by default, or `:preload true` loads all
checkpoints up front.

The CPU implementation is the numerical reference path: checkpoint values are
expanded to F32 and match the upstream Torch outputs at public precision. An
experimental fixed-shape Level Zero path keeps both ModernBERT and Laya's two
decision-head transformer layers resident. It is intentionally a lower-level
API until padded batching and the scorer are resident too:

```clojure
(require '[pretrained.decision.laya :as laya]
         '[pretrained.decision.laya-gpu :as laya-gpu])

(def built
  (laya/build-sequence (:tokenizer agent) state (:department questions)))

(def resident
  (laya-gpu/compile-decision-hidden
   agent (count (:ids built))
   {:target :ze:0 :gemm-precision :mixed-f16-f32}))

(def hidden
  (laya-gpu/decision-hidden-tokens
   resident (:ids built) (:question-type built)))

(laya-gpu/close! resident)
```

Mixed F16/F32 is the useful policy on Intel Arc; callers can select a different
Raster precision schedule for other GPUs. Cold graph instantiation remains
expensive, so compiled shapes should be cached and replayed rather than created
per request.

Downloads are sha-pinned and resume into `~/.cache/raster/models`. `HF_TOKEN` is
honoured. Passing a local directory skips download.

#### Reproducible Laya benchmark

The checked-in benchmark fixture drives the same four typed questions through
the public Clojure API and upstream Torch implementation. Pin the thread count,
report cold loading and first prediction separately, and compare the steady
medians in the emitted JSON:

```bash
MKL_NUM_THREADS=4 OMP_NUM_THREADS=4 \
clojure -M:examples:valhalla -m pretrained.laya-cpu-benchmark \
  ~/.cache/raster/models/convaiinnovations--laya 7 2

python dev/pretrained/laya_reference_benchmark.py \
  --laya-source /path/to/laya \
  --model-dir ~/.cache/raster/models/convaiinnovations--laya \
  --device cpu --threads 4 --rounds 7 --warmup 2
```

Both runners use `dev/pretrained/laya_benchmark_case.json`, verify that every
timed prediction is stable, and emit the runtime, loading time, first-call time,
warmups, individual steady samples, median, p90, and prediction. For GPU Torch
baselines use `--device xpu` or `--device cuda`; the runner synchronizes the
device around every measurement.

## Serve a local model

One call loads a checkpoint, quantizes it, opens a paged GPU worker with a
Datahike catalog, and listens on an OpenAI-compatible endpoint. Requires the
`:openai-server` alias for HTTP-kit.

```clojure
;; clojure -M:valhalla:openai-server
(require '[pretrained.openai.local :as serve])

(def server
  (serve/open-server {:model :gemma-3-270m-it        ; or :model-directory "/path"
                      :port 8080
                      :cache-directory "/var/tmp/pretrained-kv"})) ; optional, durable

;; ... later
(.close server)
```

Any OpenAI client works with only the base URL changed:

```python
from openai import OpenAI
client = OpenAI(base_url="http://127.0.0.1:8080/v1", api_key="local")
reply = client.chat.completions.create(
    model="gemma-3-270m-it",
    messages=[{"role": "user", "content": "Name the capital of France."}])
print(reply.choices[0].message.content, reply.usage.prompt_tokens_details)
```

The server applies the checkpoint's chat template for Gemma and ChatML
(Qwen, SmolLM) families and stops at its configured end-of-turn tokens. A
second turn that extends the same conversation reports the reused prefix in
`usage.prompt_tokens_details.cached_tokens`. With `:cache-directory`, chunks
and their identities persist across restarts; without it, the catalog is in
memory and removed on close. A resident conversation is advertised for reuse
at its exact end and at every chunk boundary (`:chunk-size`, default 16
tokens), so a later turn forks the shared prefix and pays only for its new
tokens. `temperature`, `top_p`, `stop`, and
`seed` are accepted but not applied; decode is greedy. See
[openai-api.md](doc/openai-api.md) for the request contract and
[serving-architecture.md](doc/serving-architecture.md) for sizing options.

## Validated model families

| Registry key | Task | Representation | Validation |
| --- | --- | --- | --- |
| `:qwen3-embedding-0.6b` (`-gpu`) | last-token embedding | 0.6B Q8 | cosine 0.999 vs Torch f32 |
| `:embeddinggemma-300m` | mean-pooled embedding | 300M Q8 GPU | cosine 0.99; 768d matryoshka |
| `:all-minilm-l6-v2`, `:bge-small-en-v1.5` | BERT embedding | 23–33M f32 | sentence-transformers parity |
| `:moonshine-streaming-medium` | streaming English ASR | 245M | LibriSpeech-100 WER 1.62%, matching reference |
| `:qwen3-asr-0.6b`, `:qwen3-asr-1.7b` | multilingual ASR | 0.6/1.7B | character-identical reference transcript |
| `:laya-english`, `:laya-multilingual`, `:laya-typed-decisions` | typed decisions | 322–421M f32 CPU | tokenizer, encoder, and typed-output parity vs Torch |
| `:gemma-3-270m-it`, `:gemma-3-1b-it` | decoder LLM | Q4/Q8 | token-exact GPU anchors |
| `:qwen3-0.6b`, `:qwen3-1.7b`, `:smollm2-135m-instruct`, `:smollm2-360m-instruct` | decoder LLM | Q4/Q8 | shared descriptor-driven engine |

Architecture support and validated registry support are different claims. The
generic loader can recognize additional compatible Hugging Face directories,
but only curated registry entries carry the validation stated above.

## Try numerical memory without a model

This command creates 512 tokens of synthetic two-layer attention state, splits
it into two immutable chunks, publishes their identities through an ephemeral
Datahike catalog, proves a repeated checkpoint is deduplicated, queries the
longest reusable prefix, and mmaps a primitive payload:

```sh
clojure -M:examples -m pretrained.numerical-memory-demo
```

After project dependencies resolve, the demo needs no model download, external
service, or GPU. Pass a directory argument to retain the local chunk files for
inspection:

```sh
clojure -M:examples -m pretrained.numerical-memory-demo /var/tmp/rstr-memory-demo
```

The demo exercises the same content identity, Konserve payload, Datahike
catalog, and mmap path used by real continuation checkpoints.

## Durable, forkable continuations

A continuation has one exact boundary: its cache contains processed positions
`[0, n)`, and its pending token is evaluated at position `n`. Checkpoints do not
need transient logits or hidden activations.

```text
token history ──> prefix hash chain ──> Datahike catalog and placement
                              │
                              └──────> immutable Konserve tensor chunks
                                                   │
                                                   └──> Raster resident pages
```

The analogy to Git is precise but bounded: chunks are immutable objects, causal
prefix hashes form parent links, and forks share unchanged pages until a write.
There is no automatic semantic merge for divergent KV caches. Read the
[continuation model](doc/continuation-model.md) for the invariants, the
publish/lookup/restore/fork lifecycle, and how it relates to LMCache and
engine prefix caches.

The model-free cluster simulator runs the same router and worker state machines
used by the runtime adapter. It makes locality/load decisions and failure traces
inspectable without a GPU or service. The database variant derives its
candidates from an exact Datahike chunk chain, ready SSD placement facts, and
ephemeral worker load/residency observations:

```clojure
;; clojure -M:examples
(require '[pretrained.continuation-controller-demo :as controller-demo])
(controller-demo/run-simulation)
(controller-demo/run-database-simulation)
```

The live variant runs the same path over two actual local Kabel WebSocket
connections, including worker heartbeats, expiry, directed offers,
acknowledgements, and fenced terminal results:

```clojure
;; clojure -M:distributed-demo
(require '[pretrained.continuation-kabel-demo :as kabel-demo])
(kabel-demo/run-live-simulation)
;; => {:selected-worker :fast-gpu,
;;     :tokens [101 102], :phase :completed, :observed-workers 2}

;; The same two worker sockets behind the OpenAI HTTP/SSE gateway:
(kabel-demo/run-openai-live-simulation)
;; => {:http-status 200, :selected-worker :fast-gpu,
;;     :text "<101><102>", :cached-token-count 0, ...}
```

It is model-free and does not send tensors through Kabel. The optional
controller middleware can be composed onto the same peers used by the
Datahike/Konserve distributed demo.

For one process, `pretrained.openai.local/open-server` (see
[Serve a local model](#serve-a-local-model)) joins the same router and worker
in memory. For several machines, `pretrained.continuation.model-worker`
assembles each worker and `dev/pretrained/real_openai_cluster_demo.clj` shows
how two of them attach to one gateway over Kabel.

The worker-side continuous-batching loop can also be inspected without model
weights. It runs variable-length prefill and decode jobs in sparse fixed Raster
lanes, retaining active lanes and priming only refills or prompt-token rows:

```clojure
;; clojure -M:examples
(require '[pretrained.continuation-batch-demo :as batch-demo])
(batch-demo/run-simulation)
```

Restore overlap has a separate deterministic REPL simulation. Its pending
restore cannot complete until the demo releases a boundary, while an unrelated
decode finishes first:

```clojure
(require '[pretrained.continuation-transfer-demo :as transfer-demo])
(transfer-demo/run-simulation)
```

A multi-request worker composes `paged-runtime/open-runtime` with
`paged/batched-handlers` and merges `paged-runtime/controller-submission` into
the worker endpoint options; the runtime owns decoder calls, so close the
controller first, then the runtime. The batch-size-one real-model smoke uses
the serialized `paged/handlers` instead. Restore, checkpoint capture, and the
per-device transfer contract are described in
[serving-architecture.md](doc/serving-architecture.md).

Workers can publish live smoothed costs instead of relying indefinitely on
deployment constants. The supplier below preserves configured values until the
corresponding runtime has produced a sample, then substitutes worker-local
prefill, first-token, and GPU restore EWMAs. It also attaches sample counts and
bounds under `:worker/live-calibration`, so policy code can distinguish a first
sample from an established estimate:

```clojure
(require '[pretrained.continuation.calibration :as calibration]
         '[pretrained.continuation.controller.kabel :as kabel]
         '[pretrained.continuation.paged-runtime :as paged-runtime])

(def live-measurements
  (calibration/measurements-fn
   runtime cache (:pool decoder)
   {:worker/prefill-ms-per-token 1.0
    :worker/first-token-ms 20.0
    :worker/gpu-restore-bytes-per-ms 1000000.0}))

(kabel/open-worker-endpoint
 (:pool decoder)
 {:worker/id worker-id
  :worker/epoch 0
  :worker/models #{model-fingerprint}}
 (merge (paged-runtime/controller-submission runtime 64)
        {:handlers handlers
         :measurements live-measurements}))
```

The calibration measures accelerator execution separately from queue delay.
This keeps hardware cost stable under load; queued demand remains an explicit
controller input rather than being counted both as latency and as backlog.
Checkpoint EWMAs are exported in the attached calibration for admission and
retention policies, but are not candidate-routing fields.

For an already loaded model, the benchmark helper separates prefill, checkpoint
submission and durability, prefix restore, uncached suffix work, first-token
latency, and context-indexed steady decode:

```clojure
(require '[pretrained.kv-continuation-demo :as demo])

(demo/run-paged-continuation-benchmark!
 model prompt-ids datahike-config "/var/tmp/gemma-kv-benchmark"
 {:max-position 2048
  :chunk-size 256
  :page-size 16
  ;; Optional fixed multi-row prompt tile; short tails use scalar decode.
  :prefill-T 32
  :decode-tokens 16
  ;; Requires page capacity for the source and an unrelated decode route.
  :checkpoint-overlap-decode-tokens 16
  :warmups 1
  :iterations 5})
```

The result reports transfer bytes and commands separately from wall time,
attributes checkpoint time to catalog lookup, device export, local persistence,
and publication, and distinguishes first process use from
process/page-cache-warm restores. It does not label warm filesystem pages as
cold SSD performance. One-time query compilation and mmap preparation happen
at manager startup and are charged to mmap preparation in the checkpoint and
cache-admission break-even rather than to the first request. The demo's `:cache-policy-calibration :worker-observation-patch` converts measured
prefill, lower-tier load, and GPU upload rates into the fields consumed by the
cluster candidate planner; merge it into that worker's observation. Its
`:checkpoint-admission` value is directly usable by paged handlers after adding
an expected reuse count:

```clojure
(require '[pretrained.continuation.controller.paged :as paged])

(def checkpoint-policy
  (assoc (:checkpoint-admission (:cache-policy-calibration result))
         :expected-reuses 2.0))

(paged/batched-handlers runtime cache decoder
                        {:checkpoint-policy checkpoint-policy})
```

The policy may instead be a function of the completed request, output, resident
route, and manager counters, allowing cluster demand to drive expected reuse.
Only a positive expected-value decision enters the bounded checkpoint queue, and
only successful catalog publication marks the resident route durable. The optional
checkpoint-overlap decode is classified as `:eligible` when Raster reports an
independent device-event transfer queue and as `:interference-only` otherwise.
Both cases retain the raw transfer and decode measurements.

For the optional two-worker S3/Kabel showcase, start an S3-compatible MinIO
service on `localhost:9000`, configure its credentials as documented in the demo
namespace, and run:

```clojure
;; clojure -M:distributed-demo
(require '[pretrained.distributed-continuation-demo :as distributed])
(distributed/run-minio-smoke!)
```

The smoke writes through a worker-local filestore to S3, publishes catalog and
placement facts through a Kabel-backed Datahike writer, promotes chunks to a
second worker, mmaps them, checks token-exact resume, and restarts that worker.

## Architecture

A model architecture is a role-to-tensor descriptor plus numerical flags such
as normalization, RoPE variant, GQA, sliding windows, and MoE routing. The generic
engine interprets the descriptor using Raster-compilable blocks.

| Namespace | Responsibility |
| --- | --- |
| `pretrained.embed`, `.asr`, `.lm` | curated task APIs and model registries |
| `pretrained.loader`, `.hub`, `.safetensors` | model dispatch, pinned downloads, tensor files |
| `pretrained.decoder`, `.decoder-gpu` | descriptor-driven CPU and resident GPU decoding |
| `pretrained.continuation.*` | chunking, catalog, placement, page pools, scheduling, restore |
| `pretrained.model-identity` | compatibility fingerprints for durable state |
| `pretrained.openai`, `.openai.server`, `.openai.local`, `.openai.cluster` | OpenAI-compatible request boundary, HTTP/SSE ingress, one-process and Kabel composition |
| `pretrained.chat`, `pretrained.continuation.model-worker` | chat templates and stop tokens; real-model worker assembly |
| `raster.*` | typed numerical compiler, schedules, kernels, buffers, and device runtime |

Linear weights repack into int8/int4 streams for Raster's CPU int8-MAC or GPU
dp4a kernels. Quantized streams are cached beside model weights: the first load
performs conversion and warm loads reuse it.

Further reading:

- [Git-like continuation model](doc/continuation-model.md)
- [Distributed paged inference architecture](doc/serving-architecture.md)
- [Cluster continuation controller](doc/cluster-controller.md)
- [OpenAI-compatible serving boundary](doc/openai-api.md)
- [Local and private model roadmap](doc/local-private-model-roadmap.md)
- [Numerical memory and simulation direction](doc/numerical-memory.md)
- [Contributing and validation](CONTRIBUTING.md)

## Validation

Model ports are compared layer-by-layer with their reference implementation,
then checked end-to-end with token, transcript, or embedding anchors. Run the
model-free suite with:

```sh
clojure -M:test
```

Model and device anchors are tagged and excluded from default CI because they
require multi-gigabyte weights or specific hardware. See
`test/pretrained/anchors_test.clj` and [CONTRIBUTING.md](CONTRIBUTING.md) before
making numerical or performance claims.

## Ecosystem direction

pretrained-rstr is an inference and numerical-memory component, not the whole
[Simmis](https://simm.is/) product. It demonstrates how replaceable model
execution can participate in a longer-lived system of immutable state, parallel
attempts, provenance, and controlled adoption. The scientific-simulation
direction uses the same idea for scenario and ensemble branches while keeping
equations, calibration, validation, and external actions explicit.

## License

Copyright © 2026 Christian Weilbach

pretrained-rstr is [MIT licensed](LICENSE). Model weights retain their own
licenses; users are responsible for complying with each model's terms.
