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
suffix text under different causal prefixes cannot alias. A compatibility
fingerprint (`model-identity/compatibility-fingerprint`) hashes the weights
identity, architecture descriptor, configuration, attention-state layout, and a
named `:execution-variant` into one value, so incompatible caches do not mix.
Name the execution variant explicitly whenever a distinct quantization or
execution scheme is in use; it defaults to `:default`, and two numerically
different builds that share a default variant would share state. Pass
`:weights-id` (for example an immutable repository revision) to avoid streaming
the weights file to compute the identity.

KV branches have no semantic merge operation. Two continuations
can fork from one prefix and progress independently; selecting one result is an
application decision. This matches Simmis's broader rule that parallel attempts
remain separate until an authorized adoption step.

## Stack responsibilities

The architecture separates three questions that a process-local cache never
has to answer, and gives each to the layer that can answer it durably.

| Question | Layer | Owns |
| --- | --- | --- |
| What is this state? | Datahike | chunk identity, parent prefix, model fingerprint, placement, demand, policy, history |
| Where are the bytes? | Konserve | immutable, content-addressed tensor chunks over local mmap or an S3-compatible store |
| Who is computing on it now? | Raster | resident page pools, sharing, copy-on-write, eviction, routed append/attention, transfers |

pretrained-rstr sits across all three: it defines the exact model boundary,
splits state into chunks, drives restore and checkpoint, and integrates the
scheduler.

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

## Retention groups

Attention state is stored per retention group: an ordered set of
`[slab layer]` members with one extent and one row width. A model without
sliding windows has one implicit token group, and its layout, fingerprint, and
chunk content ids are exactly those it had before groups existed. A model with
sliding-window layers, such as Gemma 3, has a `:global` token group for its
full-attention layers and a `:window-N` group for its sliding layers. Group
membership comes from the same predicate the decoders use to choose attention
visibility (`attention-state/global-layer?` and `layer-window`).

A chain node of a multi-group layout stores one part per group, published in
one Datahike transaction; the node carries no node-level blob, so a path that
does not understand parts fails instead of loading one part as a chunk.
Restoring the state at boundary b loads every token-group part but only the
window-group parts that intersect `[b - w + 1, b)`, the rows the next token
attends to. `pretrained.continuation.parts` makes that selection;
`pretrained.continuation.manifest` turns it into a certified Raster
`NumericalStateManifest`, and paged restore loads exactly the manifest's
`load-plan`.

## Lifecycle of a continuation

Two movements connect the layers. Publish flows down from resident pages to
durable chunks to catalog facts. Restore flows up from a catalog lookup to
verified chunks to resident pages. A fork touches only the page layer.

**Publish** is ordered so the catalog never runs ahead of the bytes:

1. capture immutable tensor ranges from the page pool;
2. make the local chunk durable;
3. wait for the authoritative backend receipt when write-behind is configured;
4. transact the chunk identity and metadata into Datahike.

The catalog never advertises a chunk that exists only in a failed worker-local
write. Capture is bounded and asynchronous, so a saturated checkpoint queue
drops speculative cache work rather than delaying tokens.

**Lookup** walks the prefix hash chain from the root and stops at the first
missing or incompatible node. The result is the longest exact prefix for which
compatible chunks exist, together with where ready replicas are placed.

**Restore** localizes each chunk through Konserve, verifies its content identity
and fingerprint, maps it into page-pool pages, and marks the replica ready. A
continuation becomes runnable only after every required page event completes.
For a window group, the restored route records a window floor, the first row
it holds; rows below it are never loaded and never read, because Raster lowers
a sliding window to the attention loop's lower bound. The fingerprint's
recorded layout must equal the target pool's layout.
Datahike is consulted for lookup and placement, not per token; workers keep
their hot scheduling state locally.

**Fork** creates a second continuation that references the source's pages,
either the whole route or the pages covering a shorter advertised boundary.
A fork whose next token would attend below a window floor is refused, and a
restored route advertises only the boundaries a fork can serve.
Nothing is copied until a branch appends into a partially filled page, at
which point that page alone is copied. A worker advertises each resident
route at its exact end and at every chunk boundary, so a request that shares
only part of a route still forks it. Two branches then advance
independently; there is no semantic merge of divergent attention state.

**Evict** removes resident pages by measured value. Dirty, pinned, protected,
or actively leased routes are excluded, and a durable route can be evicted
freely because restore can reproduce it.

## Resident execution

Durable chunks and GPU pages are intentionally different units. A 256-token
storage chunk can scatter into sixteen 16-token device pages. This lets storage
optimize for object count and sequential transfer while the executor optimizes
for allocation, prefix sharing, and attention traversal. The sizes are set when
opening components: `:chunk-size` on `manager/open-manager`, `:page-size` and
`:physical-pages` on `paged-decoder/open!`, and `:maxpos` with
`:cache-mode :paged` on `decoder-gpu/bind-decode!`.

Raster owns stable resident allocations and executable graphs. pretrained-rstr
updates compact route descriptors between submissions, allowing independent
continuations to share one fixed-capacity decode graph. Full pages can be shared
between forks; partial tails use copy-on-write. Dirty, pinned, or leased routes
are excluded from eviction.

The scheduler compares measured resident reuse, restore, and recompute costs.
Approximate repair is opt-in. Decisions carry reasons so that policy can be
inspected and, when recorded by the surrounding system, replayed.

## What is verified and how

The model-free suite (`clojure -M:test`) covers chunking and lookup, the
Datahike catalog and placement, mmap restoration and durability receipts,
checkpoint capture and publication, page-pool sharing and copy-on-write,
admission and eviction, paged append/attention plans, lane scheduling, and the
router/worker state machines. The socket-level suite runs the same machines
over live Kabel peers. Tagged GPU anchors in `test/pretrained/anchors_test.clj`
cover greedy resident decode, paged decode with a copy-on-write fork,
continuous batching, and token-exact GPU to mmap to GPU resume on real Gemma
weights. See [CONTRIBUTING.md](../CONTRIBUTING.md) for the validation levels
and [serving-architecture.md](serving-architecture.md) for the list of known
gaps.

## Related systems

Chained prefix hashing is shared with vLLM's automatic prefix caching and with
[LMCache](https://github.com/LMCache/LMCache), whose chunked token database
hashes each 256-token chunk together with its parent prefix hash. LMCache is a
KV cache management layer for datacenter engines: it offloads across GPU, CPU,
disk, and remote tiers, transfers state between prefill and decode workers, and
supports non-prefix reuse and compression. Its compatibility key is the
served model name, tensor-parallel layout, and dtype.

pretrained-rstr differs in scope and in what it treats as identity. It targets
small private models on a JVM with an integrated or modest discrete GPU, keys
compatibility by a hash of the weights, descriptor, attention layout, and
execution variant, records lineage and placement as queryable Datahike facts,
and exposes copy-on-write forking of a resident continuation to the caller.
The two are complementary rather than competing; the identity and catalog
layer here does not assume a particular storage tier.

See [serving-architecture.md](serving-architecture.md) for the distributed
executor design and the root README for runnable examples.
