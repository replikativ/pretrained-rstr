# Attention state beyond per-token KV

This note records how the continuation model extends to the model families the
field is moving to, what the surveyed serving systems do, and how the
extension stays inside the Raster north star (`../raster/design/compiler-north-star.md`) and the
durable-state contract (`../raster/design/durable-numerical-state.md`) so
that the same design serves one integrated GPU and a data-centre deployment.
Steps 1 to 4 of the [resulting order](#resulting-order) are implemented:
retention groups, per-group storage, sliding windows, the certified manifest
projection, and the structured execution variant. Per-sequence state,
tensor-parallel chunk grids, and router-emitted distributed plans (steps 5 to
7) are design only.

## Why the current model is not enough

`pretrained.attention-state/layout` describes every slab as "so many elements
per token". Chunking, page sharing, copy-on-write forks, and boundary
advertisement all follow from that. Three model families break it:

| Family | Example | Per-token state | Per-sequence state |
| --- | --- | --- | --- |
| Sliding window on most layers | Gemma 3 1B: 5 of 6 layers, window 512 | 26.6 KB/token total, 4 KB/token global | last 512 rows per windowed layer |
| Hybrid linear attention | GLM-5.3-Flash: 34 KDA + 11 sparse MLA layers; Kimi K3: 69 + 24 | 5.6 KB/token (fp8 MLA) | 140.8 MiB (GLM), 428.6 MiB (K3), length-independent, TP-sharded |
| Multi-pool sparse attention | DeepSeek V3.2/V4: latent cache + indexer cache + compressed pools; V4.1 sources KV from 4 of 40 layers | several pools with different row sizes and token strides | none |

The resolver used to accept a fixed-size slab and treat it as per-token, so a
hybrid descriptor would have allocated state × tokens. It now rejects any slab
that is not sized per token.

Recurrent state has properties the store must respect (Marconi, arXiv
2411.19379; the same rules appear in vLLM's `MambaManager`, SGLang's
`_split_node`, and vLLM's NIXL worker):

1. it is constant-sized regardless of how many tokens it represents;
2. it is updated in place, so a state cannot be rolled back to a prefix; and
3. it is orders of magnitude larger than one token's KV.

## The north-star frame

Four commitments in the north star decide the shape of the extension:

- **Durable state is a manifest, not a memory ABI.** `NumericalStateManifest`
  binds logical `AbstractValue` fields to content-addressed chunks with
  lineage, numerical compatibility, and provenance; the durable-state note
  says explicitly that "KV cache state becomes one use case, not a separate
  memory ABI". A continuation chunk is therefore a manifest whose fields have
  different extents, not a pretrained-specific record.
- **Distribution lives in the value and the plan.** `AbstractValue` carries
  placement and sharding; `DistributedPlan` owns global-to-shard mappings,
  routes, and a topology simulator that prices bytes; `WorkloadPlan` owns
  admission, continuous batching, checkpoints, and reallocation, and selects
  certified variants without injecting service policy into the IR. Rung D8 of
  the demonstrator ladder is precisely "key/value continuation state: prefix
  index over the content chain and a cross-node transfer plan", measured on
  hit rate and transfer bytes against serving systems.
- **Recompute versus checkpoint is a schedule axis** with a cost vector and
  live-range memory accounting, not a runtime heuristic.
- **Every approximation carries an error model**; exact transformations need
  none. Quantized caches, approximate reuse, and composed segments are
  proposals with bounds, never silent substitutes.

## Groups and extents

Retention is a property of layers, not slabs: Gemma 3's `:key` slab spans all
18 layers of the 270M model, of which 15 are sliding and 3 (layers 5, 11, 17)
are global. The unit is therefore a **group**: an ordered set of `[slab layer]`
members with one extent and one row width.

- `{:kind :token}` — rows per token. Paged, prefix-shareable, copy-on-write.
  GQA K/V, MLA latent, indexer caches.
- `{:kind :window :size w}` — rows per token, of which only the last `w` are
  attended to. Sliding-window layers.
- `{:kind :sequence}` — one fixed-size block per continuation, not per token.
  Recurrent and conv state of KDA/GDN/Mamba layers. Design only; `layout`
  rejects it.

A group is a manifest field shaped `[members, tokens, row-width]` (or
`[members, state-elements]` for `:sequence`), so its members share one row
width. A layout without `:groups` has one implicit token group containing every
member in payload order; the resolved layout omits `:groups` in that case, so
the attention-state layout hashed into the fingerprint, and every existing
chunk content id, is unchanged for models without windows. Groups are derived
from the same global-layer predicate the decoders use to choose attention
visibility, so group membership and the attention that reads it cannot
diverge.

## Chunk contents and identity

A chain node for `[start, end)` holds one part per group: rows for `:token`
and `:window` groups over that range and, once `:sequence` groups exist, the
snapshot *at `end`*, with `:chunk/state?` recording whether the snapshot is
present. The state at a boundary is a manifest with one field per group. Field placement and sharding are manifest facts;
the catalog stores identity, lineage, compatibility, and placement, never
bytes.

Identity does not change. The chunk hash still commits to the token ids and
the parent hash, and the compatibility fingerprint pins the model and
execution variant, so a state snapshot is a deterministic function of an
already-addressed prefix. The snapshot is content, not identity. Every
surveyed system keys the same way; none puts state into the key.

Lookup changes for hybrid layouts: the reusable prefix is the longest chain
whose last matched chunk carries state, because the store cannot derive a
state for a boundary that has none. vLLM scans right to left for the last
valid state block; SGLang's match walk accepts only nodes with a state slot.

## Fork and restore

- `:token` slabs fork as today: shared pages, copy-on-write on the first
  append into a partial page.
- `:sequence` slabs fork by copying the state slot on the device. Sharing a
  state slot is safe only until the first token is appended; after that the
  branches diverge irreversibly. vLLM and SGLang both allocate a fresh slot
  and memcpy on the compute stream.
- A fork at boundary `b` needs a snapshot at `b`: resident (a bounded ring of
  recent snapshots per lane), in a durable chunk, or reconstructed by
  replaying `[c, b)` from the nearest earlier snapshot `c` through the
  recurrent layers only, while the attention rows are reused unchanged.
  Replay is linear in `b - c` (Sparse Prefix Caching, arXiv 2605.05219).
  Which boundaries carry snapshots and which are replayed is the same
  checkpoint/recompute schedule the north star uses for adjoints: legality is
  exactness, feasibility is live memory, ranking is analytic then measured
  against the observed overlap distribution.
- Cross-node restore and fork are transfer steps in a `DistributedPlan`, so
  state bytes, routes, and contention with hot traffic are priced by the
  topology simulator rather than by per-router constants. At GLM-5.3-Flash
  size a fork moves 140 MiB; the durable-state note already requires the
  outer planner to schedule checkpoint traffic against fabric traffic.
- Publication order is unchanged, with one addition: a boundary's state must
  not be advertised before its copy has completed. vLLM refuses a same-step
  hit on a block whose copy-on-write has not run; our pending-append fence on
  routes is the same guard.

## Advertisement and policy

`mark-exact-prefix!` already records every chunk boundary of a resident
route. Each boundary gains `:state?`, and candidate planning for a hybrid
fingerprint matches only state-bearing boundaries. The router's estimate
adds the state copy bytes and any replay tokens, both taken from the plan's
cost vector.

Admission and eviction need size-normalized value, not recency:

- Marconi admits state only at branch points and at the end of generation, and
  evicts by recency plus FLOPs saved per byte, which is roughly 200× denser
  for state than for attention rows. SGLang keeps two LRU lists and refreshes
  only the consumed state node on a hit so one session cannot pin a whole
  path. Both allow evicting an interior node's state while keeping its rows
  (a tombstone).
- `pretrained.continuation.residency` already ranks routes by expected saved
  compute against reload cost and sharing. Extending it to per-slab bytes and
  to tombstoning state on interior boundaries is the same evaluator with two
  more inputs. It is a `WorkloadPlan` policy: it selects among certified
  variants and never enters the IR.

At the scale of one integrated GPU the datacentre constraint is loose. A
1B-class hybrid's state is a few megabytes per sequence, so snapshotting every
chunk boundary is affordable and the policy can start as "every boundary" and
tighten by measurement. The design is one; the policy is a schedule.

## Compatibility fingerprint, second tier

The fingerprint already hashes the weights identity, descriptor,
configuration, attention-state layout, and a named execution variant. Two
changes make it match both the durable-state note's restore-compatibility
list (dtype, endian, layout, coordinate system, codec, checksum domain) and
what the field learned the hard way:

1. Make the execution variant structured rather than a keyword: per slab
   group a reference to a Raster quantized-format descriptor (block shape,
   scale encoding, decode computation, tolerance) rather than a free dtype
   pair, the page layout family, positional configuration (RoPE scaling), and
   a format version. vLLM's offload tier currently collides silently on layout
   (issue #55904), on model revision (#49261), and on RoPE scaling (#56311);
   its NIXL handshake hashes model, dtype, heads, layers, backend, cache dtype,
   and speculative config, and validates parallelism and block size at
   runtime. That two-tier split is the right shape, and it must not create a
   second format registry beside Raster's.
2. Keep the chunk hash token-only. Precision belongs in the fingerprint, and
   recurrent state has its own precision axis: every engine defaults it to
   fp32 while attention KV is routinely fp8, and mixed per-layer schemes are
   shipping (vLLM `--kv-cache-dtype-skip-layers`, ModelOpt
   `MIXED_PRECISION`, Z.ai's INT8/FP8/BF16 cache). A single dtype field is
   insufficient; a per-group descriptor is not.

Sampler state, RNG state, and numerical mode stay on the request, as the
durable-state note requires for exact restart; they are not attention state.

## Raster

- Attention IR already has what sparse and quantized attention need: a CSR
  page route with CSR visibility for top-k selection, and a
  `{:dtype :quantization}` format contract on K and V pages that declines
  until its ordered ABI lands. Latent MLA slabs are `:token` slabs with
  different feature keys. The first compiler work is the quantized page ABI,
  expressed as a format descriptor in the sense of north-star §4 (int8
  per-token-head with scales on the dp4a path; the XMX path is f16 in, f32
  accumulate, and there is no fp8 dtype). A quantized cache is an
  approximation and carries an error model and golden vectors like every
  other format.
- Linear attention is not a new primitive. A chunked delta rule is a
  segmented contraction inside each chunk and an ordered fold over chunks,
  which is a `TypedStructuredControl` fixpoint with a loop-carried tensor:
  the state. The north star already states that general recurrences carry an
  ordered association contract and must not borrow the scan reassociation
  proof, which is exactly the property a delta rule needs. The per-lane state
  crosses decode steps through `ProgramStage` anchors as pages do, as a
  batch-indexed resident buffer with retained transfer events for snapshot
  and restore. Attention itself is a segmented weighted reduction over a
  route; neither it nor the recurrence adds a decoder or cache primitive to
  the compiler.
- Composition is the research lane and needs an error model. HYPIC (arXiv
  2607.01299) caches each segment's cumulative transition operator with its
  end state, so independently cached segments compose in constant time at a
  1.71-point quality gap. For a delta rule the operator is one d×d matrix per
  head, the same size as the state. It is the only published route to
  recurrent chunks that concatenate like content-addressed objects, and under
  the north star it is a fidelity transform: legal only with its bound in the
  cost vector, never a silent substitute for exact replay.

## Representing per-sequence state: tensor field, not entity kind

Two readings were considered. A distinct `AbstractValue` kind for per-entity
state (shards as entity-id sets with an ownership function, fork as a
redistribution step) matches north-star §6 and §3.9 wording, but none of it is
landed: `AbstractValue` kinds are `:tensor`, `:record`, `:opaque`;
`DistributedPlan` shards are rectangular `ValueShard`s with `:partitioned` or
`:replicated` kinds; `NumericalStateManifest` v1 fields are tensors on a
regular chunk grid; redistribution and all-to-all-v steps are designed only.

Version 1 therefore uses the tensor reading, structured so the entity reading
can be adopted later without changing durable bytes:

- The durable field is the snapshot itself, shape `[heads d d]` per layer (plus
  conv state), sharded on the head axis under tensor parallelism with the
  landed `{:kind :partitioned :axis n}` vocabulary. The lane index is
  placement, not semantics, and never enters the manifest.
- Resident state is one `[lanes ...]` buffer per layer, a whole-buffer
  `ProgramStage` state anchor like the key and value caches. A continuation
  maps to a lane through a slot table, exactly as a route maps a continuation
  to pages. Fork and snapshot are routed block moves through the existing
  `gather-blocks!`/`scatter-blocks!` route, which is already the page
  transfer engine; no new primitive.
- Cross-node fork or restore is a `DistributedPlan` transfer step over the
  sub-rectangle shard of that lane, priced by the topology simulator.
- The field carries `:attributes {:entity :continuation}` and its state
  identity in the catalog, so that when entity-set shards land, the same
  content addresses can be re-described under the entity kind and fork can
  become a redistribution step.

### Implementation of steps 1 to 4

**Validation (step 1).** `layout` rejects a slab that is not sized per token, a
token axis other than `:position`, and any group extent other than `:token` and
`:window`. In a model with sliding windows every slab must also be indexed by
model layer, and a window must span at least two tokens. Groups partition every
`[slab layer]` member exactly once and have one row width.

**Storage (step 2).** A layout with one group stores the same blob as before
groups existed, with the same content id. A multi-group layout stores one blob
per group per token range; the blob carries `:chunk/group` and a payload plan
restricted to that group's members. The catalog node for a chain position is
unchanged (fingerprint, prefix hash, parent, start, count); a multi-group node
owns one part per group (`group`, `store-key`, `bytes`) as a component, carries
no node-level blob, and is published in one transaction with all its parts,
so a node is visible only when every part is durable. A part's id is derived
from the node id and a canonical coordinate map `{:group g}`, so a shard
coordinate can be added later without renaming existing parts. Consumers read
parts through `parts/entry-parts`, which returns the single implicit part of a
single-group node, so no consumer branches on the layout. Paths that cannot
handle several parts yet, such as replica promotion, refuse multi-part nodes
as failed actions instead of loading one part as a chunk.

The catalog records once per fingerprint the resolved layout (canonical EDN),
numerical mode, and determinism. The first publication writes it with a
compare-and-swap from nil in the same transaction as its chunks. When two
first publishers race, the second swap fails and rolls back its chunks; the
publisher then re-reads the record and publishes its chunks if the layout is
equal, or fails if it differs. The content-address algorithm is recorded per
node rather than per fingerprint, so a later algorithm can publish under an
existing fingerprint. Shard and chunk-grid facts never enter the recorded
layout; they belong in part coordinates. The first publisher owns
these facts: a later publication under the same fingerprint with a different
layout, storage dtype, or contract fails. In particular a paged FP16 publisher
cannot join a fingerprint an FP32 CPU or contiguous-GPU publisher recorded,
so the cache storage format belongs in the execution variant.

**Windows (step 3).** Every group part is published for every chunk, because
each boundary is a reusable state. The saving is in restore and transfer, not
storage. The token at boundary `b` attends rows `[max(0, b - w + 1), b)` of a
window group, so a restore of `b` loads token-group parts for all chunks and
window-group parts only for the chunks intersecting that range. The route's
window floor is the start of the first loaded window chunk,
`C * floor(max(0, b - w + 1) / C)`. Rows below the floor are never read,
because Raster lowers a window to the attention loop's lower bound
(`attention-begin = max(q - window_left - kv_start, 0)`) rather than to a
mask; the CPU decoder likewise starts its loop at `kv-start`. One predicate in
`attention-state` (`global-layer?`, `layer-window`, `first-attended-row`)
serves group derivation, the decoders, part selection, and the floor checks.

The route is allocated with its floors before any row lands. A fork, a
boundary advertisement, and the local controller's reuse of a resident prefix
are refused when the next token would attend below a floor. A capture of a
window chunk below the floor fails loudly instead of exporting unwritten rows;
it cannot happen for chunks captured after a restore, because the floor is at
or before the start of the chain's tail chunk and later captures start there
or after it. The paged path applies window selection. The contiguous GPU path restores
every part, and because its decoder does not apply sliding windows, contiguous
decode and restore refuse sequences longer than the smallest window. CPU
restore works on whole snapshots rather than chunks, and the CPU decoder
applies the window.

At 4,096 tokens a restore moves 20.4 MB instead of 75.5 MB for Gemma 3 270M
and 28 MB instead of 109 MB for Gemma 3 1B. Changing Gemma's layout changes
its fingerprint, so existing Gemma caches miss rather than restore under the
new format.

**Manifest (step 2) and contract (step 4).** `continuation-manifest` projects
the catalog chain at a boundary onto a Raster `NumericalStateManifest`, one
field per group, and Raster's certifier checks it. Its id is the catalog node
id of the boundary (`catalog/chunk-entry-id`, which covers the fingerprint and
the prefix hash), its parent is the previous boundary's node, and its logical
coordinate is the processed count. A window field is shaped
`[members, b - floor, row-width]` and records the floor as `:token-origin` in
its coordinate space. Paged restore, plain and overlapped, loads the manifest's `load-plan`, so the
certified manifest decides what a boundary contains. The router prices every
boundary with the same selection in one linear pass over the chain, charging a
tier's fixed cost once per loaded part. An
execution variant may be a map `{:name :cache :numerical}`; its numerical
contract is recorded with the fingerprint's layout and reaches the manifest.
Keyword variants hash exactly as before.

### Findings from a REPL spike against Raster's certifiers

Each point below was checked by constructing the value and running Raster's
own validator, not by reading docstrings.

1. **A continuation state is a certified manifest today.** Real chunks for a
   600-token CPU continuation at chunk size 256 project onto one field shaped
   `[slabs x layers, tokens, row-width]` (`[4 600 8]`) with grid `[4 256 8]`
   and a clipped `[4 88 8]` tail, and `numerical-state/certify` accepts it.
   The existing blob payload is ordered slab, layer, token rows
   (`payload-plan`), which is exactly row-major for that field, so one stored
   chunk is one grid cell with no re-encoding. The manifest id is the
   boundary's catalog node id, `parents` is the previous boundary's node,
   `compatibility-id` is the model fingerprint, and `logical-coordinate` is the
   processed count; the pending token belongs to the request, not the state.
   The catalog therefore stays bespoke, because it indexes
   prefix lookup and placement that the manifest does not, and a manifest is a
   *projection* of catalog facts at a boundary. What the catalog lacked for
   that projection was the layout per fingerprint (field shapes and dtype,
   then only inside each blob), the content-address algorithm (implicit Hasch
   v3), and the numerical contract's mode and determinism from the structured
   execution variant. Step 2 added the layout and contract as a
   per-fingerprint record and the algorithm as a per-node fact.
2. **All three extents fit schema v1, so no partial-coverage schema is needed.**
   A window of 300 at 600 tokens is a complete field `[4 344 8]` with
   `{:token-origin 256}` (the first attended row is 301): the origin is
   rounded down to a chunk boundary, which
   retains at most C-1 extra rows and keeps origins aligned across boundaries
   (300, 600, 856, 1100 give origins 0, 256, 512, 768), so a window chunk has
   one content address in every state whose window covers it. A per-sequence
   field `[layers heads d d]` is a single-cell grid. A manifest holding a token
   field, a window field, and a sharded per-sequence field certifies. A
   boundary is state-bearing exactly when its manifest contains every field.
3. **The durable unit is field x token range, not token range.**
   `cpu-tensor-chunk` packed all slabs into one blob. Retaining Gemma's global
   layers for all chunks and its windowed layers for the last 512 tokens needs
   separate blobs per slab group of uniform row width. That is the boundary vLLM
   calls KV cache groups and LMCache calls object groups.
4. **Tensor-parallel degree belongs in the chunk grid, not the fingerprint.**
   A token field sharded on its row axis certifies with grid `[4 256 8/tp]`:
   3, 6, and 12 chunks at TP 1, 2, 4, identical logical bytes, one chunk per
   rank per token range. State published at one degree is never silently
   reused at another; cross-degree reuse is an explicit regrid. The same
   applies to per-sequence state sharded on its head axis.
5. **A cross-worker restore is a certified, priced plan today.** A 136 MiB KDA
   state sharded over two ranks restores to a third worker as two parallel
   `transfer-step`s over separate links; `simulate` and `certify` report
   5.7 ms and 142,606,336 bytes. Two constraints and two gaps: transfer
   endpoints must be mesh devices, so the mesh is the set of candidate workers
   and the value's sharding names a subset of it; transfer `:bytes` are caller
   asserted rather than derived from a shard rectangle as `HaloExchange` does;
   and a transfer claims no memory at its destination, so peak memory omits the
   restored state. The last two are Raster-side additions for rung D8.
6. **State needs a routed slot table, not a lane-indexed buffer.** The paged
   decoder gives inactive lanes append slot `-1` and an empty route so they
   mutate nothing. Per-sequence state needs the same property, so the
   recurrent step reads and writes state through `slot-table int[B]` and skips
   `-1`; a free slot is then safe to restore into during decode. A slot pool is
   the page pool with one page per sequence: restore is the existing
   block-transfer scatter into a slot index, and fork shares the slot by
   refcount and copies it on the first write with `gpu/copy-range!`, which is
   what `copy-page!` already does for shared partial tails. Forks that are
   never decoded never copy. No ranged views are needed.

### Resulting order

Implemented:

1. Reject slabs that are not per-token and extents other than `:token` and
   `:window` in `layout`.
2. Store one blob per group per token range, record the layout and numerical
   contract per fingerprint and the content-address algorithm per node in the
   catalog, and add `continuation-manifest`, a catalog-to-manifest projection
   certified by Raster that paged restore loads from.
3. `:extent :window` with chunk-aligned floors and trailing-chunk restore
   (Gemma 3).
4. Structured execution variant carrying numerical mode and determinism.

Design only:

5. `:extent :sequence` on a slot pool with a routed slot table and
   copy-on-first-write.
6. Chunk grids aligned to tensor-parallel shards, and per-rank placement facts,
   as a shard coordinate in part ids.
7. Router-emitted `DistributedPlan`s for restore and fork (rung D8), after
   Raster derives transfer bytes from shards and charges destination memory.

## What the surveyed systems chose

| System | State representation | Snapshot granularity | Fork | Keyed by |
| --- | --- | --- | --- | --- |
| vLLM V1 | one state page per block slot; attention block inflated to match | block boundary; sub-block prefill checkpoints | explicit copy before the forward | token hash + group id; no dtype, TP, or block size |
| SGLang | one state slot per radix node | max(kernel chunk, page size); rounds down | copy on match; state cannot be split, interior becomes tombstone | chained SHA-256 of tokens; backend suffix with model and parallelism; no dtype |
| TensorRT-LLM | state blocks under a sentinel window size | configured interval; reuse off unless a snapshot policy is set | whole-block copy; partial reuse disabled | chained non-cryptographic mix; connectors refused for hybrids |
| LMCache MP | opaque re-viewed page per object group with `recurrent_state` | one block | store-level copy | `ObjectKey` with group id, no dtype; codec header carries model, revision, RoPE, layout |
| Dynamo KVBM | none | — | — | xxh3 chain; salt is a free-form string |
| pretrained-rstr (proposed) | `:sequence` manifest field per lane; snapshot in chunk at boundary | any chunk boundary, by schedule | copy state slot, share token pages; cross-node as a plan step | token hash chain + structured fingerprint |

## Sources

- Raster design: `compiler-north-star.md` §3.7, §3.8, §3.9, §4, §6, §8 (D8),
  §10; `durable-numerical-state.md`.
- Marconi: Prefix Caching for the Era of Hybrid LLMs, arXiv 2411.19379.
- Sparse Prefix Caching for Hybrid and Recurrent LLM Serving, arXiv 2605.05219.
- HYPIC: Position-Independent Caching for Hybrid-Attention LLMs, arXiv 2607.01299.
- vLLM `vllm/v1/kv_cache_interface.py`, `vllm/v1/core/single_type_kv_cache_manager.py`,
  `vllm/distributed/kv_transfer/kv_connector/v1/nixl/`, main at 2026-09-18.
- SGLang `python/sglang/srt/mem_cache/mamba_radix_cache.py` and
  `unified_cache/components/mamba.py`, main at 2026-09-18.
- LMCache `lmcache/v1/kv_layer_groups.py`,
  `lmcache/integration/vllm/kv_cache_group_edits.py`, and
  docs.lmcache.ai/mp/hybrid_models.html, at 2026-09-18.
- Model cards: `zai-org/GLM-5.3-Flash`, `moonshotai/Kimi-K3`,
  `deepseek-ai/DeepSeek-V4-Flash`, `deepseek-ai/DeepSeek-V4.1-Flash`.
