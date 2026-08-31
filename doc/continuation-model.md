# Git-like continuations for language-model inference

pretrained-rstr treats the expensive state behind an LLM response as a durable,
forkable value. A continuation is not a transcript and it is not a serialized
process. It is the minimum exact boundary needed to resume decoding: processed
token history, the next pending token, a model/execution fingerprint, and the
attention-state tensors produced so far.

## The boundary

For a continuation with `processed-count = n`:

- attention state contains positions `[0, n)`;
- `pending-token` is evaluated at position `n` by the next step;
- the token history includes the pending token;
- no logits or transient hidden activations need to be persisted.

This boundary is shared by CPU, contiguous GPU, and paged GPU execution. Restore
therefore resumes the same causal computation rather than reconstructing an
approximation from text.

## What “Git-like” means here

The analogy is structural, not a claim that a KV cache can be merged like text.

| Git concept | Inference continuation |
| --- | --- |
| blob | immutable tensor chunk in Konserve/object storage |
| object id | Hasch content identity stored as a Datahike `store-ref` |
| commit parent | previous token-prefix chunk hash |
| tree/path metadata | Datahike facts describing model, range, placement, and tier |
| branch | a logical continuation sharing immutable prefix pages |
| checkout | restore chunks into a worker-local Raster page pool |
| copy-on-write | share full prefix pages; copy a partial tail before mutation |

The prefix chain commits to both a chunk's token ids and its parent hash. Equal
suffix text under different causal prefixes cannot alias. A model fingerprint
also pins architecture, weights, and execution-relevant representation, so
incompatible caches do not silently mix.

KV branches do not currently have a semantic merge operation. Two continuations
can fork from one prefix and progress independently; selecting one result is an
application decision. This matches Simmis's broader rule that parallel attempts
remain separate until an authorized adoption step.

## Stack responsibilities

```text
Datahike     identity, prefix lineage, placement, demand, policy, history
    │
Konserve     immutable, content-addressed tensor chunks; local mmap or S3
    │
Raster       resident page pools, routed append/attention, transfers, execution
    │
pretrained   exact model boundary, chunking, restore, scheduling integration
```

Datahike does not store every float as a datom and is not queried for each decode
step. Workers maintain their hot scheduling state locally. Datahike records the
decision-grade control plane: what state exists, what it is compatible with,
where a replica is ready, and which worker requested it. Tensor bytes follow a
separate data plane through mmap-compatible Konserve stores or an authoritative
object store.

Publication is ordered deliberately:

1. capture immutable tensor ranges;
2. make the local chunk durable;
3. wait for the authoritative backend receipt when write-behind is configured;
4. transact the chunk identity and metadata into Datahike.

The catalog never advertises a chunk that exists only in a failed worker-local
write. Restore verifies the content identity and compatibility before making a
replica ready.

## Resident execution

Durable chunks and GPU pages are intentionally different units. A 256-token
storage chunk can scatter into sixteen 16-token device pages. This lets storage
optimize for object count and sequential transfer while the executor optimizes
for allocation, prefix sharing, and attention traversal.

Raster owns stable resident allocations and executable graphs. pretrained-rstr
updates compact route descriptors between submissions, allowing independent
continuations to share one fixed-capacity decode graph. Full pages can be shared
between forks; partial tails use copy-on-write. Dirty, pinned, or leased routes
are excluded from eviction.

The scheduler compares measured resident reuse, restore, and recompute costs.
Approximate repair is opt-in. Decisions carry reasons so that policy can be
inspected and, when recorded by the surrounding system, replayed.

## Current implementation status

Implemented and covered by model-free tests:

- content-addressed token-prefix chunks and longest-prefix lookup;
- Datahike catalog, demand, placement, and replica state;
- local mmap restoration and tiered-store durability receipts;
- bounded asynchronous checkpoint capture/publication;
- resident page allocation, sharing, copy-on-write, admission, and eviction;
- routed paged append/attention plans and fixed-capacity lane scheduling;
- transfer and continuation latency instrumentation.

GPU/model anchors additionally cover exact resume and Gemma paged decode on the
supported development hardware. Mixed prefill/decode packing, native CUDA copy
streams, and production cluster policy remain engineering work rather than
finished product claims.

See [serving-architecture.md](serving-architecture.md) for the distributed
executor design and the root README for runnable examples.
