# pretrained-rstr

[![Clojars Project](https://img.shields.io/clojars/v/org.replikativ/pretrained-rstr.svg)](https://clojars.org/org.replikativ/pretrained-rstr)
[![CircleCI](https://circleci.com/gh/replikativ/pretrained-rstr.svg?style=shield)](https://circleci.com/gh/replikativ/pretrained-rstr)
[![Slack](https://img.shields.io/badge/slack-join_chat-brightgreen.svg)](https://clojurians.slack.com/archives/C09622F337D)

> ⚠️ **Experimental**: pretrained-rstr is under active development. APIs may change before 1.0. Feedback welcome!

Run pretrained HuggingFace models — **text embeddings, speech-to-text, and decoder
LLMs** — natively on the JVM, on the [raster](https://github.com/replikativ/raster)
typed-dispatch compiler. No Python, no ONNX runtime: weights load from safetensors,
quantize to int8/int4 streams, and run on raster's CPU int8-MAC kernels and
GPU-resident Level Zero/OpenCL programs.

```clojure
(require '[pretrained.embed :as emb]
         '[pretrained.asr :as asr]
         '[pretrained.lm :as lm])

;; every modality is the same shape: (load-X :key [dir] [opts]) then a task verb.
;; a registry key auto-downloads weights from HF on first use; an explicit local dir
;; skips the download (bring your own weights for a known model).

;; embeddings
(def e (emb/load-embedder :qwen3-embedding-0.6b))
(emb/embed-texts e ["Datahike is a durable Datalog database."])
;; => {:data float[n*1024] :n 1 :dim 1024}   (L2-normalized rows)

;; speech-to-text — any audio format (wav pure-JVM; mp3/ogg-opus/m4a via ffmpeg)
(def m (asr/load-asr :moonshine-streaming-medium))
(asr/transcribe m "voice-note.oga")
;; => "And so my fellow Americans, ask not what your country can do for you, ..."
(asr/transcribe m "talk.wav" {:timestamps? true})
;; => {:text "..." :words [{:word "And" :start 0.0 :end 0.02} ...]}

;; decoder LLMs
(def g (lm/load-lm :gemma-3-270m-it))                ;; or (lm/load-lm :gemma-3-270m-it {:gpu? true})
(lm/generate-text g "The capital of France is" 20)
;; => " Paris. ..."
```

## Models

| Registry key | Task | Size | Quality (validated) |
|---|---|---|---|
| `:qwen3-embedding-0.6b` (+`-gpu`) | embeddings, last-token | 0.6B Q8 | cos 0.999 vs torch f32 |
| `:embeddinggemma-300m` | embeddings, bidirectional + mean pool | 300M Q8 (GPU) | cos 0.99 vs torch; 768d matryoshka |
| `:all-minilm-l6-v2`, `:bge-small-en-v1.5` | embeddings (BERT tier) | 23–33M f32 | parity with sentence-transformers |
| `:moonshine-streaming-medium` | English ASR, **true streaming** | 245M | **WER 1.62% == HF torch** (LibriSpeech-100); word timestamps |
| `:qwen3-asr-0.6b` / `-1.7b` | multilingual ASR (52 languages) | 0.6/1.7B | transcript char-identical to torch gold |
| `:gemma-3-270m-it` / `:gemma-3-1b-it` | decoder LLM | ≤1B Q4/Q8 | token-exact GPU decode vs oracle; CPU ≈ llama.cpp speed |
| `:qwen3-0.6b` / `:qwen3-1.7b`, `:smollm2-135m-instruct` / `:smollm2-360m-instruct` | decoder LLMs | ≤1.7B Q4/Q8 | same descriptor-driven engine (shared attention/norm stack) |

Embeddings feed directly into [proximum](https://github.com/replikativ/proximum)
(`emb/rows` → HNSW vector index) and [umap-rstr](https://github.com/replikativ/umap-rstr)
(`emb/flat-doubles` → 2-D layouts). BERT-family sentence encoders (MiniLM/bge, mean-pool)
run self-contained in `pretrained.arch.bert` — no extra dependency; `:engine :encoder`
registry entries route there automatically.

## How it works

A model architecture is a *descriptor* — a role→tensor-name map plus ~10 flags (norm
type/gain, rope variants, GQA, qk-norm, sliding windows, sandwich norms, MoE routing)
— interpreted by one generic engine over raster's compilable `deftm` blocks. Adding a
standard decoder-LM is a descriptor, not engine code.

- `pretrained.embed` / `pretrained.asr` / `pretrained.lm` — task-level APIs, all the same
  shape: a curated registry + HF auto-download + `load-X`/task-verb (CPU or `{:gpu? true}`)
- `pretrained.decoder` — the descriptor-driven decode engine (`load-hf`, `decode-step`,
  `generate-cached`); GPU-resident decode/prefill in `pretrained.decoder-gpu`
- `pretrained.loader` — the low-level generic loader: architecture registry, tokenizer
  auto-detection, `from-pretrained` (dispatch any HF dir on `config.json` model_type — the
  advanced, unvalidated path behind the curated `pretrained.lm` registry)
- `pretrained.arch.*` — decoder descriptors as pure data: `gemma3`, `llama`, `qwen3`,
  `qwen3-moe`, `embedding-gemma`, plus the self-contained BERT encoder `bert`;
  `pretrained.asr.*` — moonshine, qwen3-asr
- `pretrained.tokenizer.{sp,bpe,wordpiece}` — HF `tokenizer.json` tokenizers
- `pretrained.hub` — sha-pinned downloads with resume + sha256 into
  `~/.cache/raster/models` (`HF_TOKEN` honored; every `load-*` also accepts a local dir)
- `pretrained.safetensors` / `pretrained.audio` — format frontends (bf16/f16 fast paths;
  WAV pure-JVM, other audio via ffmpeg)

**Quantized execution:** linear weights repack into int8/int4 streams executed by
raster's spin-pool int8-MAC kernel (CPU) or GPU dp4a kernels. Q8_0 is measured
lossless for embeddings; decode uses Q4. Quantized streams are disk-cached next to the
weights — the first load quantizes once (~30s for 0.6B), warm loads take ~5s.

## Durable attention-state cache

Decoder inference can checkpoint immutable, prefix-hashed attention-state chunks
without blocking the generation thread. Datahike indexes logical identity and
placement policy; Konserve/Boring stores contiguous FP32 payloads; restore mmaps
one chunk at a time and uploads its slices directly through Raster. The final
prompt token remains pending, so restoring a prefix resumes at exactly the same
continuation boundary as uninterrupted inference.

```clojure
(require '[pretrained.continuation.benchmark :as bench]
         '[pretrained.continuation.manager :as cache]
         '[pretrained.model-identity :as identity])

(def fingerprint
  (identity/compatibility-fingerprint
   model {:execution-variant {:backend :ze
                              :linear-weights :q4k
                              :attention-state :float32}}))

(def manager
  (cache/open-manager
   {:store {:backend :memory :id (random-uuid)}
    :schema-flexibility :write :keep-history? false :value-caps :default}
   "/var/tmp/pretrained-kv"
   {:chunk-size 256}))

;; dstate is a bound pretrained.decoder-gpu state; prompt-ids is a token vector.
(bench/benchmark-gpu-prefix! manager dstate fingerprint prompt-ids
                             {:warmups 1 :iterations 5})
```

The benchmark separates prefill, checkpoint submission/drain, first measured
restore, and process/page-cache-warm restore; it does not label warm pages as
cold SSD. Chunk identities use Hasch and are published as Datahike
`:db.type/store-ref` values, while the prefix hash separately identifies the
causal token chain.

For cluster durability, pass an already connected authoritative Konserve store
(for example Konserve-S3) as `:chunk-backend-store`. The manager owns its local
filestore but the caller retains ownership of the backend connection:

```clojure
(cache/open-manager datahike-config "/var/tmp/pretrained-kv"
                    {:chunk-size 256
                     :chunk-backend-store s3-store})
```

Chunk checkpoints write the local mmap-compatible frontend first. Their
`:captured` future can therefore complete while the remote copy is in flight;
`:published` completes only after every Konserve write-behind receipt succeeds
and the Datahike transaction commits. Datahike never advertises a store-ref that
only exists in one worker's cache.

`pretrained.continuation.placement` records declarative per-worker demands and
observed replicas. `pretrained.continuation.replica/open-executor` connects that
control plane to a bounded background copy worker: it moves bytes off-band,
verifies the immutable Hasch identity and catalog metadata, then transitions the
target replica through `copying` to `ready` or `failed`. Transaction listeners
only offer work and never perform I/O. The built-in promoter uses Konserve's
tiered store to explicitly synchronize one content key from its authoritative
backend into the worker's local filestore frontend. It waits for that write and
verifies the frontend before publishing `ready`; restore then mmaps the concrete
frontend, not the tiered wrapper. The current tier sync decodes once during
promotion. Remote or raw-copy transports can implement the narrow
`ReplicaPromoter` effect without replacing Konserve's storage API, and Yggdrasil
can version the catalog through its Datahike adapter.

The [distributed paged inference design](doc/serving-architecture.md) separates
durable transfer chunks from GPU allocation pages and specifies the path to
continuous batching, asynchronous restore/checkpoint streams, and explainable
admission, prefetch, and eviction policy.

With MinIO listening on `localhost:9000`, the optional cluster alias runs a
self-cleaning, model-free end-to-end check. It writes through a worker-local
filestore to S3, commits catalog and placement facts through a Kabel writer,
promotes the chunks to a second worker, mmaps them, and restarts that worker:

```clojure
;; clojure -M:distributed-demo
(require '[pretrained.distributed-continuation-demo :as distributed])
(distributed/run-minio-smoke!)
```

`open-authority!`, `open-worker!`, `checkpoint-gpu-prefix!`, and
`restore-gpu-prefix!` expose the same stages for a bound model. The source phase
returns a small serializable manifest for the destination phase. Concurrent
workers should run in separate JVMs; the smoke turns them over sequentially
because distributed-scope intentionally keeps one in-process route per remote
peer.

```clojure
(require '[konserve.tiered :as tiered]
         '[pretrained.continuation.placement :as placement]
         '[pretrained.continuation.replica :as replica])

(def worker-b-tiered
  (tiered/connect-tiered-store
   worker-b-store shared-store
   :write-policy :frontend-only
   :read-policy :frontend-first
   :opts {:sync? true}))

(def executor
  (replica/open-executor
   connection "worker-b" :ssd
   (replica/konserve-tiered-promoter worker-b-tiered)))

(placement/request!
 connection {:model-fingerprint fingerprint :prefix-hash prefix
             :node "worker-b" :tier :ssd :priority 10})
```

### Resident paged attention

`pretrained.continuation.page-pool` maps the same durable chunks into stable
worker-local Raster allocations. A logical page covers every attention-state
slab and layer, so full prefix pages can be shared safely while a partial tail
uses copy-on-write. Durable chunk size and device page size are independent: a
256-token Konserve object can scatter into sixteen 16-token GPU pages.

`pretrained.continuation.paged-attention` binds those pools to Raster's semantic
`AttentionProblem` and verified `KernelGraph`. Fixed-capacity runners update
packed query positions and dense page tables between submissions, allowing
unrelated continuations to form one batch without changing graph pointers.

`pretrained.continuation.paged-append` reserves one physical slot per batch
lane and binds projected resident FP32 K/V rows directly to Raster's
routed FP16 assignment graph. The same reservation batch is reused across
layers and becomes visible only after every layer write succeeds. Attention may
lease its prospective post-append route and queue behind assignment on the same
in-order Raster session without publishing unfinished state.

`pretrained.continuation.scheduler` plans bounded continuous batches with decode
lanes first and chunked repair/prefill work in the remaining capacity. It also
chooses between measured resident, restore, recompute, and explicitly enabled
approximate-repair paths. `pretrained.continuation.residency` admits routes under
GPU pressure with explainable cost-aware eviction; dirty, pinned, and leased
routes are never victims.

```clojure
(require '[pretrained.attention-state :as attention-state]
         '[pretrained.continuation.manager :as cache]
         '[pretrained.continuation.page-pool :as pages]
         '[pretrained.continuation.paged-attention :as paged-attn])

(def pool
  (pages/open-pool!
   (:sess dstate)
   (attention-state/layout (:model dstate))
   {:page-size 16 :physical-pages 1024 :dtype :half}))

(cache/restore-paged-prefix!
 manager pool :request-a fingerprint prompt-ids
 {:admit? true
  :policy {:durable? true :reuse-probability 0.6
           :recompute-ms 18.0 :reload-ms 5.0 :last-access 42}})

(def runner
  (paged-attn/open-runner!
   pool {:layer 0 :batch-size 8 :total-query-tokens 8
         :q-heads 4 :kv-heads 1 :qk-head-dim 64 :value-head-dim 64
         :pages-per-sequence 128}))
```

For model execution, `:query-view` and `:output-view` may be FP16 or FP32 Raster
`ResidentBufferView`s owned by adjacent projection and output graphs. The runner
then uploads only small route descriptors and returns the resident output view
after completion; query and attention tensors never cross the host.

`pretrained.continuation.paged-decoder` replaces the contiguous K/V assignment
and attention interval between generated pre-attention and post-attention
stages. The adapter declares K/V as stage state and the attention result as its
output; Raster's operation-neutral `ProgramStage` derives and validates the
interval from executable effects. One `LinkPlan` then interleaves every layer's
generated pre-stage, routed append graph, paged-attention graph, and generated
post-stage before linking the token head/tail. A decode step uploads one shared
set of route descriptors and replays this composite executable once. Pretrained
owns page allocation, reservations, leases, and transactional publication, but
neither scans kernel ABI names nor assembles raw compiler phase keys. Bind with
`:cache-mode :paged` to omit the displaced per-layer K/V and score buffers:

```clojure
(require '[pretrained.loader :as loader]
         '[pretrained.decoder-gpu :as decoder]
         '[pretrained.continuation.page-pool :as pages]
         '[pretrained.continuation.paged-decoder :as paged]
         '[raster.gpu.core :as gpu])

(def model (loader/from-pretrained "/models/gemma-3-270m-it"))
(def dstate (decoder/bind-decode! model :maxpos 64 :cache-mode :paged))
(def engine (paged/open! dstate :page-size 16 :physical-pages 128))
(def tokenizer (:tokenizer model))
(def prompt (vec ((:encode tokenizer) (:tok tokenizer)
                  "The capital of France is")))

(try
  (let [tokens (paged/generate! engine :request-a prompt 4)]
    ;; => [9079 236764 532 506], decoded as " Paris, and the"
    (pages/fork-route! (:pool engine) :request-a :request-b)
    tokens)
  (finally
    (paged/close! engine)
    (gpu/close-session! (:sess dstate))))
```

The contiguous decoder is likewise one validated LinkPlan containing every
layer plus the head and greedy tail. Both modes import the same pretrained-owned
resident allocations without copying them or transferring ownership. Routed
graph temporaries stay private to their descriptor steps; query, K/V, attention
output, page slabs, and route descriptors are stable linked nodes.

The Gemma anchor verifies token-exact parity with the contiguous decoder, the
absence of `kc*`, `vc*`, and `sc` allocations, shared-page fork semantics, and
copy-on-write resume parity. Partial pages copy directly between resident
Raster views (Level Zero unified allocation copy or OpenCL device-buffer copy),
without JVM tensor staging.

The routed attention leaf is still a portable correctness reference and this
first model executor has one decode lane. The scheduler and attention/append
adapters already describe multi-lane batches, but widening generated projection
and post-attention stages for continuous model batching remains performance
work. Durable chunks restore through `restore-paged-prefix!`. Newly generated
paged routes enter the same Hasch-chain/Konserve/Datahike pipeline through
`checkpoint-paged-chunks-async!`: capture leases an immutable route snapshot,
gathers arbitrary physical page spans on the bounded worker, and publishes only
after tiered-store durability. Scheduling when to request that optional capture
remains a serving-policy decision.

## Validation methodology

Every port is validated against its reference implementation before it ships:
layer-by-layer activation comparison vs HF transformers golds (typical agreement
~1e-6 relative in f32), then end-to-end anchors — token-exact decode, character-exact
transcripts, cos ≥ 0.999 embeddings vs torch f32. The anchors are repeatable tests:

```
clojure -M:test        # fast, model-free unit tests
# with local weights (see test/pretrained/anchors_test.clj):
clojure -A:dev:test:valhalla -M -e "(require 'clojure.test 'pretrained.anchors-test) \
  (clojure.test/run-tests 'pretrained.anchors-test)"
```

## Requirements

- JDK 21+ (Panama FFI); the raster Valhalla toolchain for full performance
  (see raster's README for the JVM flags)
- OpenBLAS for the f32 GEMM paths
- ffmpeg (optional, for non-WAV audio)
- GPU (optional): Level Zero for Intel, or a compatible OpenCL ICD for Intel/NVIDIA/AMD

## License

Copyright © 2026 Christian Weilbach

pretrained-rstr is [MIT licensed](LICENSE). Model **weights** carry their own licenses
(all registry models are Apache-2.0/MIT) — you are responsible for complying with each
model's terms.
