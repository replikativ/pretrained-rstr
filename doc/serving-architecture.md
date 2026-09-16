# Distributed paged inference

This document describes the serving architecture as implemented: the units of
storage and execution, which component owns each job, how the scheduler and
cluster controller fit together, the Raster execution contract, and the
restore and checkpoint flows. Design direction that is not yet implemented is
collected at the end so a reader can tell the two apart.

The durable cache is one input to scheduling, not the scheduler itself.
Datahike holds queryable durable facts and desired placement; each inference
process owns its latency-critical GPU allocator, queues, and transfer streams.
For the memory model itself (identity, durable bytes, resident pages, and the
publish/lookup/restore/fork lifecycle) see
[continuation-model.md](continuation-model.md); this document does not restate
it.

## Units of storage and execution

Three granularities serve different workloads and remain independent:

| Unit | Typical size | Purpose |
| --- | ---: | --- |
| GPU page | 16–32 tokens per slab | allocation, sharing, copy-on-write, eviction |
| Durable chunk | 128–512 tokens | hashing, object-store transfer, catalog publication |
| Request span | arbitrary | one logical continuation and scheduler lane |

A durable chunk contains an integral sequence of serialized slab ranges and may
fill several GPU pages. Its final range may end in a partial page. Page size is
a runtime/kernel choice; changing it does not change the Hasch content identity
of a durable chunk. Conversely, changing durable chunk size changes prefix-chain
nodes but not the attention result.

The sizes are set when opening components: `:chunk-size` on
`manager/open-manager`; `:page-size` and `:physical-pages` on
`paged-decoder/open!`; `:maxpos`, `:cache-mode :paged`, and `:batch-size` on
`decoder-gpu/bind-decode!`. Physical pages bound the worker's resident
capacity; `:maxpos` bounds one continuation's context.

The causal hash chain is the authoritative exact-prefix index. A chunk commits
to its parent hash and token range, so a lookup can stop safely at the first
missing or incompatible node. Datahike can also index token ranges, owners,
tenants, models, time, and observed reuse; the chain is not the only query
path.

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

`pretrained.continuation.scheduler` is the pure iteration planner. Each plan is
replayable: decode lanes consume one token first, then bounded repair and
prefill chunks consume the remaining token and sequence budgets. Its phases are
decode, prefill, and repair. Admission is constrained by a token budget, free
GPU pages, model/layout compatibility, and estimated restore-versus-recompute
cost. Decode receives priority to avoid inter-token latency spikes, while a
bounded prefill budget prevents starvation. Cache-source selection is explicit;
modular repair is ineligible unless the caller supplies both an opt-in and a
minimum quality threshold.

Each request record contains its model fingerprint, token history, processed
count, pending token, deadline/priority, and logical page table. Sampling state
belongs to a request, not to reusable attention state: many samplers may share
the same exact prefix pages.

A single-lane decoder bound with `:prefill-T` executes one complete multi-row
prompt tile per scheduler turn. Decode remains higher priority and may run
before the next tile; an incomplete tail uses the ordinary one-row lane graph.
This bounds the latency a prompt can impose on active decode by the selected
tile rather than by the whole prompt. Multi-lane prefill shares the decode graph
one row per lane.

### Cluster controller

The cluster lifecycle is implemented in `pretrained.continuation.controller.*`.
A pure router ranks request-specific exact-prefix candidates by predicted time
to first token and fences retries by assignment attempt. A worker-local
interpreter repeats admission against current device state before accepting,
then invokes the manager and paged decoder through injected handlers. The same
pure machines run in a deterministic failure simulator; see
[cluster-controller.md](cluster-controller.md).

Candidate derivation batches exact chunk and ready-replica queries against the
local Datahike snapshot, then combines them with versioned ephemeral worker
observations. It retains GPU, RAM/SSD/object prefix boundaries, and recompute
as explicit alternatives, so a stale GPU location can fall back without
discarding the worker's durable cache.

The optional Kabel adapter carries these observations and the fenced assignment
lifecycle over live connections. Heartbeats are connection-scoped and expire
without Datahike transactions; catalog and placement facts remain durable and
queryable. The adapter passes unrelated messages through, so this control plane
can share the Kabel connection used by a worker's Datahike replica.

Workers can publish live smoothed costs instead of deployment constants.
`calibration/measurements-fn` preserves configured values until the runtime has
produced a sample, then substitutes worker-local prefill, first-token, and GPU
restore averages, with sample counts attached so policy can distinguish a first
sample from an established estimate. Accelerator execution is measured
separately from queue delay, so hardware cost stays stable under load and
queued demand remains an explicit controller input rather than being counted
both as latency and as backlog.

### GPU cache manager

One manager per device owns physical pages and never delegates allocation to
Datahike. A page moves through `free`, `loading`, `resident`, `evicting`, and
`free`; a generation counter prevents a late transfer event from completing
into a reused page. Resident pages have refcounts and immutable prefix
identity. Appending to a shared partial tail uses copy-on-write.

The manager exposes reservations, page-table installation, pin/unpin, and
asynchronous load/evict operations. A request becomes runnable only after its
required page events complete; unrelated lanes continue decoding.

`pretrained.continuation.paged-runtime` is the device-local execution loop. It
accepts controller restore, prefill, and decode jobs through cancellable
futures, uses `scheduler/plan-iteration` for each graph step, and maps selected
mixed-phase work onto stable physical lanes. Prompt rows are primed every
prefill step; retained decode rows reuse the embedding emitted by the decoder
tail. Inactive fixed lanes receive no route and append no page. Restores run
outside that loop: a bounded Konserve `ContentProvider` localizes chunks,
scoped mmap leases are retained by Raster upload events, and handlers poll
completion between unrelated graph submissions. A continuation becomes
runnable only after every chunk event completes. Fragmented restore issues
direct range batches.

`pretrained.continuation.residency` is the deterministic admission and
eviction evaluator over page-pool snapshots. It ranks durable routes by
expected saved compute, lower-tier reload cost, sharing/SLO bonuses, and
recency. Planning simulates shared-page refcounts; application revalidates
under the pool lock and cannot evict dirty, pinned, protected, or actively
leased routes.

Admission reserves the full projected prompt-plus-generation page demand.
Existing resident pages are credited and shared partial tails include a
possible copy-on-write page. Cancellation fences logical work immediately but
does not release physical capacity until an accepted local operation has
quiesced.

### Raster execution contract

Attention state is a set of physical slab pools described by
`pretrained.attention-state/layout`. Standard GQA has key and value pools;
sliding-window models add per-layer retention metadata; MLA can describe latent
and rotary slabs. Kernels receive page geometry and slab bindings rather than
assuming contiguous `kcN`/`vcN` arrays.

The pretrained adapter binds FP16 query and output `ResidentBufferView`s
directly into routed attention, so projection, attention, and output graphs
share allocations without tensor uploads or downloads. The decoder declares the
cache writes and attention output as a Raster `ProgramStage`; Raster selects
the unique effect-defined interval and projects ordinary
before/selected/after descriptors. Each routed append and attention
`KernelGraph` is wrapped as an ordinary descriptor instance and interleaved
with every layer's before/after descriptors and the head/tail in one validated
`LinkPlan`, so each token is one linked replay. Page reservation, prospective
leases, and transactional publication stay outside the graph, so a failed
replay cannot publish a partially written page. Cache management therefore
stays outside the compiler without an attention-specific linker or decoding
ABI names in the runtime.

Raster provides the routed append operation with this explicit ordered ABI:

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
queue. `pretrained.continuation.paged-append` binds resident projection views
to that graph. An append batch holds the page-manager reservations across all
layers; `pretrained.continuation.paged-attention` can pin the corresponding
prospective routes while their writes are queued. Only successful completion
of every layer event commits the batch. Page allocation, copy-on-write,
generation checks, and route mutation remain cache-manager responsibilities
rather than kernel semantics.

Continuous batching is a host scheduling operation: completed lanes leave,
newly ready lanes enter, and the same compiled Raster programs consume a
different descriptor on the next replay.

## Restore and checkpoint flows

Restore is planned before bytes move:

1. Query the longest exact compatible prefix and its observed locations.
2. Reserve enough GPU pages or choose a smaller prefix under pressure.
3. Prefer already-resident shared pages, then local SSD, RAM, peer, object
   store, or recomputation according to measured completion cost.
4. Stream durable chunks through bounded mmap scopes into reserved page ranges.
   Chunk boundaries do not need to align with pages.
5. Publish local `ready` observations only after content verification; mark
   the request runnable only after GPU transfer events complete.
6. Prefill the uncached suffix and leave the final token pending.

Raster retains each Konserve mmap lease through its asynchronous upload event.
The worker polls completion between decode iterations and exposes the restored
route only after all required chunks are resident.

`checkpoint-paged-chunks-async!` checkpoints immutable completed page ranges.
Capture uses the symmetric retained download API: its event pins the source
route until the host payload is complete, and the manager never awaits an
incomplete event on the decoder thread. The capture worker writes each host
payload before allocating the next, bounding staging to one durable chunk;
`:max-chunk-staging-bytes` rejects an oversized optional capture before it
enters the queue. Konserve write-behind copies to S3 and Datahike publication
waits for backend receipts. If either queue is full, the optional checkpoint is
skipped rather than delaying tokens.

Manager startup prepares the exact Datahike chunk query and the scoped
Konserve/Boring tensor mapping before publishing local readiness, so one-time
query compilation and FFM initialization are not charged to the first inference
request.

Physical copy/compute overlap is a per-device property that Raster reports
through `page-pool/transfer-capabilities`. OpenCL reports an independent
physical transfer queue; Level Zero shared-memory downloads complete inline.
Independent queues make live overlap eligible, and the benchmark classifies an
overlapped run as `:eligible` or `:interference-only` on that basis; measured
event and decode timings still determine whether a device benefits under load.

## Policy

Policy has a fast local evaluator and a durable control plane. The scheduler
uses a periodically refreshed immutable snapshot and process-local
measurements. It does not query Datahike for every batch.

Admission scores the alternatives `resident reuse`, `SSD/S3 restore`, and
`recompute` using predicted queue delay plus transfer or prefill time. Eviction
ranks unpinned pages by recomputation cost, lower-tier availability, sharing,
and recency, and respects active leases and protected routes; absence of a
demand is not permission to delete durable data. Checkpoint admission is a
positive expected-value decision over measured capture cost and expected
reuse, and only successful catalog publication marks a resident route durable.

Policy is deterministic weighted cost with recorded inputs and reasons, so a
decision can be inspected and, where the surrounding system records it,
replayed.

## Datahike facts

Immutable chunk, demand, and replica entities are the catalog. Kabel routes all
catalog mutations through one authoritative writer, and its sync stream lets
each worker update a local Datahike replica and react to relevant
transactions. Tensor bytes remain off-band in Konserve-S3 and worker-local
filestores, avoiding a second trip through the catalog writer. Store refs
express content identity and reachability; placement determines which workers
pull the referenced object. High-rate telemetry is aggregated locally;
Datahike stores decision-grade facts, not every kernel timestamp.

## Known gaps

These are not implemented. The docs above do not depend on them.

- Preemption and resumption of an admitted request. The scheduler has no
  preempted queue.
- Streaming chunks directly into page reservations without host staging, and a
  staged scatter pipeline for fragmented restore.
- A dedicated low-priority transfer stream and bounded pinned-memory pool to
  control copy-engine and memory-bandwidth contention with inference.
- Native CUDA copy streams; CUDA is reached today only through OpenCL.
- Recording each candidate snapshot, selected alternative, predicted cost, and
  measured outcome in Datahike for offline replay.
- A fixed-shape multi-sequence prefill graph; multi-lane prefill shares the
  decode graph one row per lane.
- Layout adapters and oracle tests for sliding/global attention retention and
  MLA latent slabs in the paged path.
- Measured multi-process throughput and interference numbers. The library
  makes exactness claims, not throughput claims.

## Design direction

The following describes intended shape rather than shipped behaviour.

Requirements the design is held to: preserve the exact continuation boundary;
share exact prefixes without copying resident pages; admit, preempt, resume,
and mix prefill and decode work without a model-sized contiguous cache per
request; keep checkpoint, object-store, catalog, and policy work off the
inference stream; support CUDA and Level Zero through Raster's backend-neutral
buffer, stream, and event interfaces; describe persistent attention state as
named slabs so standard, sliding-window, and latent state share one storage and
policy engine; and keep decisions explainable from Datahike history without
putting Datalog or network round trips in the per-token path.

The paged decoder is intended to accept one descriptor for a ragged batch
rather than one recorded graph per request. The shipped binding keys are
`:page-table`, `:lengths`, and `:start-positions` for attention and
`:slot-mapping` for append; a single unified batch descriptor is not yet
defined.

Simmis can later optimize policy weights or replace the evaluator, with every
decision and outcome stored as facts for offline replay. Yggdrasil supplies the
versioned state/history boundary. Proximum is useful for candidate generation:
find similar sessions, predict likely next prefixes, and cluster reuse
observations. Approximate matches never directly reuse attention state; a
candidate still passes exact tokens, model fingerprint, layout, position
semantics, and causal chain verification. Approximate KV reuse is a separate
research feature with an explicit quality contract.

The contiguous decoder remains the correctness oracle and a useful small-model
path. Paging is introduced behind a separate execution interface rather than
rewriting it in place.
