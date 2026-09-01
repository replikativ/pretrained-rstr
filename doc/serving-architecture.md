# Distributed paged inference

This document defines the serving architecture that the durable continuation
prototype should grow into. Its two goals are to make pretrained models useful
as a continuously batched inference runtime and to expose a concrete integration
of Raster, Datahike, Konserve, Simmis, and Proximum.

The durable cache is one input to scheduling, not the scheduler itself. Datahike
holds queryable durable facts and desired placement; each inference process owns
its latency-critical GPU allocator, queues, and transfer streams.

## Requirements

- Preserve the exact continuation boundary: attention state covers
  `[0, processed-count)`, and `pending-token` is processed next.
- Share exact token prefixes without copying their resident GPU pages.
- Admit, preempt, resume, and mix prefill and decode work without allocating a
  model-sized contiguous KV cache per request.
- Keep checkpoint, object-store, catalog, and policy work off the inference
  stream. Saturation drops speculative cache work, not tokens.
- Support CUDA/NVIDIA and Level Zero through Raster's backend-neutral buffer,
  stream, and event interfaces.
- Describe persistent attention state as named slabs. Standard K/V, sliding
  window K/V, and latent attention state must not require different storage or
  policy engines.
- Make decisions explainable and replayable from Datahike/Yggdrasil history,
  while keeping Datalog and network round trips out of the per-token hot path.

## Units of storage and execution

Three granularities serve different workloads and must remain independent:

| Unit | Typical size | Purpose |
| --- | ---: | --- |
| GPU page | 16–32 tokens per slab | allocation, sharing, copy-on-write, eviction |
| Durable chunk | 128–512 tokens | hashing, object-store transfer, catalog publication |
| Request span | arbitrary | one logical continuation and scheduler lane |

A durable chunk contains an integral sequence of serialized slab ranges and may
fill several GPU pages. Its final range may end in a partial page. Page size is a
runtime/kernel choice; changing it does not change the Hasch content identity of
a durable chunk. Conversely, changing durable chunk size changes prefix-chain
nodes but not the attention result.

The existing causal hash chain remains the authoritative exact-prefix index. A
chunk commits to its parent hash and token range, so a lookup can stop safely at
the first missing or incompatible node. Datahike can also index token ranges,
owners, tenants, models, time, and observed reuse; the chain is not the only
query path.

## Components and ownership

```text
                       Datahike / Kabel writer
                 catalog, demand, leases, observations
                                |
                    tx reports / policy snapshots
                                v
  request ---> scheduler ---> local cache manager ---> Raster batch executor
                 |                 |        |             | GPU page pool
                 |                 |        +-- SSD mmap --+ transfer stream
                 |                 +----------- RAM staging
                 v
          admission/prefetch policy
                 |
                 +---------- Konserve tiered store ---------- S3
                              immutable chunks
```

### Scheduler

The scheduler owns request lifecycle and builds a new execution batch each
iteration. It maintains decode, prefill, restore-ready, transfer-waiting, and
preempted queues. Admission is constrained by a token budget, free GPU pages,
model/layout compatibility, deadlines, and estimated restore-versus-recompute
cost. Decode requests normally receive priority to avoid inter-token latency
spikes, while a bounded prefill budget prevents starvation.

Each request record contains its model fingerprint, token history, processed
count, pending token, sampling state, deadline/priority, and logical page table.
Sampling state is part of a resumable request but not part of reusable attention
state: many samplers may safely share the same exact prefix pages.

The first planner implementation is `pretrained.continuation.scheduler`. Its
iteration plan is pure and replayable: decode lanes consume one token first,
then bounded repair and prefill chunks consume the remaining token and sequence
budgets. Cache-source selection is also explicit; modular repair is ineligible
unless the caller supplies both an opt-in and a minimum quality threshold.

The cluster lifecycle is implemented separately in
`pretrained.continuation.controller.*`. A pure router ranks request-specific
exact-prefix candidates by predicted TTFT and fences retries by assignment
attempt. A worker-local interpreter repeats admission against current device
state before accepting, then invokes the manager and paged decoder through
injected handlers. The same pure machines run in a deterministic failure
simulator; see [cluster-controller.md](cluster-controller.md).

Candidate derivation batches exact chunk and ready-replica queries against the
local Datahike snapshot, then combines them with versioned ephemeral worker
observations. It retains GPU, RAM/SSD/object prefix boundaries, and recompute as
explicit alternatives, allowing a stale GPU location to fall back without
discarding the worker's durable cache.

The optional Kabel adapter now carries these observations and the fenced
assignment lifecycle over live connections. Heartbeats are connection-scoped
and expire without Datahike transactions; catalog and placement facts remain
durable and queryable. The adapter passes unrelated messages through so this
control plane can share the existing Kabel connection used by a worker's
Datahike replica.

### GPU cache manager

One manager per device owns physical pages and never delegates allocation to
Datahike. A page moves through `free`, `loading`, `resident`, `evicting`, and
`free`; a generation counter prevents a late transfer event from completing into
a reused page. Resident pages have refcounts and immutable prefix identity.
Appending to a shared partial tail uses copy-on-write.

The manager exposes reservations, page-table installation, pin/unpin, and
asynchronous load/evict operations. Transfers use a dedicated CUDA/Level Zero
stream and events. A request becomes runnable only after its required page events
complete; unrelated lanes continue decoding.

`pretrained.continuation.paged-runtime` is the device-local execution loop. It
accepts controller restore, prefill, and decode jobs through cancellable futures,
uses `scheduler/plan-iteration` for each graph step, and maps selected mixed-phase
work onto stable physical lanes. Prompt rows are explicitly primed every prefill
step; retained decode rows reuse the embedding emitted by the decoder tail.
Inactive fixed lanes receive no route and append no page. Current restores run
outside that loop: a bounded Konserve `ContentProvider` localizes chunks, scoped
mmap leases are retained by Raster upload events, and handlers poll completion
between unrelated graph submissions. A continuation becomes runnable only
after every chunk event completes. Fragmented restore currently favors direct
range batches; a staged scatter pipeline remains a measured optimization.

`pretrained.continuation.residency` now implements the first deterministic
admission evaluator over page-pool snapshots. It ranks durable routes by expected
saved compute, lower-tier reload cost, sharing/SLO bonuses, and recency. Planning
simulates shared-page refcounts; application revalidates under the pool lock and
cannot evict dirty, pinned, protected, or actively leased routes.

Admission now reserves the full projected prompt-plus-generation page demand.
Existing resident pages are credited and shared partial tails include a possible
copy-on-write page. Cancellation fences logical work immediately but does not
release physical capacity until an accepted local operation has quiesced.

### Raster execution contract

The paged decoder should accept one descriptor for a ragged batch rather than
one recorded graph per request:

```clojure
{:tokens       int[B]
 :positions    long[B]
 :sequence-len long[B]
 :page-table   int[B,max-pages]
 :page-count   int[B]
 :slot-offset  int[B]}
```

Attention state is a set of physical slab pools described by
`pretrained.attention-state/layout`. Standard GQA has key and value pools;
sliding-window models add per-layer retention metadata; MLA can describe latent
and rotary slabs. Kernels receive page geometry and slab bindings rather than
assuming contiguous `kcN`/`vcN` arrays. The first implementation can bucket by
model, dtype, page geometry, and decode/prefill mode; heterogeneous model batches
are not required.

Continuous batching then becomes a host scheduling operation: completed lanes
leave, newly ready lanes enter, and the same compiled Raster programs consume a
different descriptor on the next replay.

The pretrained adapter can already bind FP16 query and output
`ResidentBufferView`s directly into routed attention, so projection, attention,
and output graphs can share allocations without tensor uploads or downloads.
The current single-lane decoder declares the cache writes and attention output
as a Raster `ProgramStage`; Raster selects the unique effect-defined interval
and projects ordinary before/selected/after descriptors. Pretrained wraps each
routed append and attention `KernelGraph` as an ordinary descriptor instance,
then interleaves those instances with every layer's before/after descriptors and
the head/tail in one validated `LinkPlan`. Each token therefore needs one linked
replay, not four submissions per layer plus a head submission. Page reservation,
prospective leases, and transactional publication remain outside the graph, so
a failed replay cannot publish a partially written page. This keeps cache
management outside the compiler without introducing an attention-specific
linker or decoding compiler ABI names in the runtime.

Raster 0.2.355 provides the routed append operation with this explicit ordered
ABI:

```clojure
{:k-rows       fp32[B,n-kv,head-dim]
 :v-rows       fp32[B,n-kv,value-head-dim]
 :slot-mapping int[B] ; physical-page * page-size + page-offset
 :k-pages      fp16[physical-pages,page-size,n-kv,head-dim]
 :v-pages      fp16[physical-pages,page-size,n-kv,value-head-dim]}
```

It converts projected rows with round-to-nearest-even and assigns each lane to
its unique reserved slot. Its graph effect reads rows and slot mapping and
writes both page pools, so Raster orders append before attention on the same
queue. `pretrained.continuation.paged-append` binds resident projection views to
that graph. An append batch holds the page-manager reservations across all
layers; `pretrained.continuation.paged-attention` can pin the corresponding
prospective routes while their writes are queued. Only successful completion of
every layer event commits the batch. Page allocation, copy-on-write, generation
checks, and route mutation remain cache-manager responsibilities rather than
kernel semantics.

## Restore and checkpoint flows

Restore is planned before bytes move:

1. Query the longest exact compatible prefix and its observed locations.
2. Reserve enough GPU pages or choose a smaller prefix under pressure.
3. Prefer already-resident shared pages, then local SSD, RAM, peer, object store,
   or recomputation according to measured completion cost.
4. Stream durable chunks through bounded mmap scopes into reserved page ranges.
   Chunk boundaries do not need to align with pages.
5. Publish local `ready` observations only after content verification; mark the
   request runnable only after GPU transfer events complete.
6. Prefill the uncached suffix and leave the final token pending.

`checkpoint-paged-chunks-async!` checkpoints immutable completed page ranges.
The bounded capture worker submits validated direct downloads across arbitrary
physical page spans. Raster retains the route lease through device completion,
while the worker polls without holding the decoder session. It writes each host
payload before allocating the next, bounding staging to one durable chunk.
Konserve write-behind copies to S3 and Datahike publication waits for backend
receipts. If either queue is full, the optional checkpoint is skipped. A
dedicated low-priority transfer stream and bounded pinned-memory pool remain
needed to control copy-engine and memory-bandwidth contention with inference.

## Policy

Policy has a fast local evaluator and a durable control plane. The scheduler uses
a periodically refreshed immutable snapshot and process-local measurements. It
does not query Datahike for every batch.

Admission scores the alternatives `resident reuse`, `SSD/S3 restore`, and
`recompute` using predicted queue delay plus transfer or prefill time. Prefetch
uses known routing, session affinity, active placement demands, and prefix reuse
frequency. Eviction ranks unpinned pages by recomputation cost, next-use
probability, size, age, and lower-tier availability. It must respect active
leases, tenant quotas, and a protected working-set floor; absence of a demand is
not permission to delete durable data.

Initial policy should be deterministic weighted cost with recorded inputs and
reasons. Simmis can later optimize weights or replace the evaluator, with every
decision and outcome stored as facts for offline replay. Yggdrasil supplies the
versioned state/history boundary.

Proximum is useful for candidate generation: find semantically or structurally
similar sessions, predict likely next prefixes, and cluster reuse observations.
Approximate matches must never directly reuse attention state. A candidate still
passes exact tokens, model fingerprint, layout, position semantics, and causal
chain verification. Approximate KV reuse is a separate research feature with an
explicit quality contract.

## Datahike facts

The current immutable chunk, demand, and replica entities are a valid base. The
serving system adds short-lived entities for request intent, worker/device
capacity, leases, transfer observations, cache hits, recompute measurements, and
policy decisions. High-rate raw telemetry should be aggregated locally before
transaction; Datahike stores decision-grade facts, not every kernel timestamp.

Kabel routes all catalog mutations through one authoritative writer. Its sync
stream lets each worker update a local Datahike replica and react to relevant
transactions. Tensor bytes remain off-band in Konserve-S3 and worker-local
filestores, avoiding a second trip through the catalog writer. Store refs express
content identity and reachability; placement determines which workers pull the
referenced object.

## Delivery order

1. Exercise the existing chunks and placement logic with a Kabel writer, two
   worker-local filestores, and a shared S3-compatible store. Measure cold, warm,
   partial, restart, and failed-transfer behavior.
2. Introduce backend-neutral page geometry, page tables, and a model-free GPU
   page allocator with refcount, generation, and lease tests.
3. Add Raster paged decode for a homogeneous batch, then paged prefill and mixed
   continuous batches. Compare every path with the existing contiguous oracle.
4. Stream chunks directly into page reservations and add preemption/resumption.
5. Implement deterministic admission, prefetch, and eviction policy with recorded
   explanations; then connect Simmis optimization and Proximum candidates.
6. Add layout adapters and oracle tests for Gemma sliding/global attention and
   DeepSeek MLA before enabling cross-model production serving.

The current contiguous decoder remains the correctness oracle throughout. It is
also a useful small-model path; paging should be introduced behind a separate
execution interface rather than rewriting it in place.
