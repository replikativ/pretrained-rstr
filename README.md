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

Use JDK 21 or newer and add the latest released library to `deps.edn`:

```clojure
{:deps {org.replikativ/pretrained-rstr {:mvn/version "0.1.26"}}}
```

The source tree currently targets Raster `0.2.425`. OpenBLAS is required for
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
(def gpu-model (lm/load-lm :gemma-3-270m-it {:gpu? true}))
```

Downloads are sha-pinned and resume into `~/.cache/raster/models`. `HF_TOKEN` is
honoured. Passing a local directory skips download.

## Validated model families

| Registry key | Task | Representation | Validation |
| --- | --- | --- | --- |
| `:qwen3-embedding-0.6b` (`-gpu`) | last-token embedding | 0.6B Q8 | cosine 0.999 vs Torch f32 |
| `:embeddinggemma-300m` | mean-pooled embedding | 300M Q8 GPU | cosine 0.99; 768d matryoshka |
| `:all-minilm-l6-v2`, `:bge-small-en-v1.5` | BERT embedding | 23–33M f32 | sentence-transformers parity |
| `:moonshine-streaming-medium` | streaming English ASR | 245M | LibriSpeech-100 WER 1.62%, matching reference |
| `:qwen3-asr-0.6b`, `:qwen3-asr-1.7b` | multilingual ASR | 0.6/1.7B | character-identical reference transcript |
| `:gemma-3-270m-it`, `:gemma-3-1b-it` | decoder LLM | Q4/Q8 | token-exact GPU anchors |
| Qwen3 and SmolLM2 registry entries | decoder LLM | Q4/Q8 | shared descriptor-driven engine |

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
[continuation model](doc/continuation-model.md) for the invariants and stack
ownership.

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
```

It is model-free and does not send tensors through Kabel. The optional
controller middleware can be composed onto the same peers used by the
Datahike/Konserve distributed demo.

The worker-side continuous-batching loop can also be inspected without model
weights. It runs variable-length prefill and decode jobs in sparse fixed Raster
lanes, retaining active lanes and priming only refills or prompt-token rows:

```clojure
;; clojure -M:examples
(require '[pretrained.continuation-batch-demo :as batch-demo])
(batch-demo/run-simulation)
```

A single-lane decoder bound with `:prefill-T` executes one complete multi-row
prompt tile per scheduler turn. Decode remains higher priority and may run before
the next tile; an incomplete tail uses the ordinary one-row lane graph. This
bounds preemption latency by the selected tile rather than by the whole prompt.
Multi-lane prefill continues to share the decode graph one row per lane until a
fixed-shape multi-sequence prefill graph is available.

Restore overlap has a separate deterministic REPL simulation. Its pending
restore cannot complete until the demo releases a boundary, while an unrelated
decode finishes first:

```clojure
(require '[pretrained.continuation-transfer-demo :as transfer-demo])
(transfer-demo/run-simulation)
```

Real workers construct `paged-runtime/open-runtime`, pass
`paged/batched-handlers` to their local/Kabel controller, and merge
`paged-runtime/controller-submission` into the worker endpoint options. The
runtime owns decoder calls; close the controller first, then the runtime.
Raster retains each Konserve mmap lease through its asynchronous upload
event. The worker polls completion between decode iterations and exposes the
restored route only after all required chunks are resident. Checkpoint capture
uses the symmetric retained download API: its event pins the source route until
the host payload is complete, while the capture worker writes at most one staged
chunk at a time. `:max-chunk-staging-bytes` rejects an oversized optional capture
before queueing. Raster reports the physical transfer contract per worker:
OpenCL has an independent transfer queue, while Level Zero shared-memory capture
currently copies inline. Independent queues make live overlap eligible; measured
event and decode timings still determine whether a device benefits under load.

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
cold SSD performance. Manager startup warms the exact Datahike chunk query so
query compilation cannot extend the first checkpoint's route lifetime. The
bounded checkpoint and localization workers likewise prepare and validate the
scoped Konserve/Boring tensor mapping before publishing local readiness. This
moves one-time FFM/navigation initialization off the first inference request
without copying or decoding tensor elements; the benchmark charges it to mmap
preparation in the checkpoint and cache-admission break-even. The
demo's `:cache-policy-calibration :worker-observation-patch` converts measured
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
| `raster.*` | typed numerical compiler, schedules, kernels, buffers, and device runtime |

Linear weights repack into int8/int4 streams for Raster's CPU int8-MAC or GPU
dp4a kernels. Quantized streams are cached beside model weights: the first load
performs conversion and warm loads reuse it.

Further reading:

- [Git-like continuation model](doc/continuation-model.md)
- [Distributed paged inference architecture](doc/serving-architecture.md)
- [Cluster continuation controller](doc/cluster-controller.md)
- [OpenAI-compatible serving boundary](doc/openai-api.md)
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
