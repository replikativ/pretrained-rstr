# Cluster continuation controller

The controller assigns an exact continuation request to one inference worker,
but it does not centralize GPU allocation or tensor traffic. Cluster routing is
advisory; the selected worker is authoritative for its device memory.

```text
OpenAI-style request
        |
        v
cluster router -- offer/timeout --> worker-local controller
        ^                              | reserve pages atomically
        |                              | restore -> prefill -> decode
        +------ accepted/result -------+
                    Kabel/Netz control messages

Datahike: model, lineage, placement, and recorded decision-grade facts
Konserve: immutable tensor chunks over local mmap, peers, or S3
Raster:   device pages, transfers, continuous batches, model execution
```

## Implemented boundary

`pretrained.continuation.controller.router` and `.worker` are pure state
machines. They emit effect maps and do no I/O. Assignment ids contain the
request id and monotonically increasing attempt; late timeouts and results from
failed workers therefore cannot complete a newer attempt.

`pretrained.continuation.controller.sim` interprets the same machines under
logical time. Tests cover locality/load trade-offs, stale capacity, directed
partitions, worker crashes, timeout fallback, cancellation, and deterministic
replay. Numerical work is represented by declared costs and token values, so
these failure cases run without a GPU or model.

`pretrained.continuation.controller.cluster` adds real offer timers and a
consumer delivery callback. `pretrained.continuation.controller.local` wraps a
real page pool. Before sending an accepted reply it:

1. validates worker epoch and model compatibility;
2. computes incremental prompt-plus-maximum-generation page demand;
3. protects the active route and plans eligible durable evictions;
4. reserves the pages atomically;
5. invokes injected restore, prefill, and decode handlers.

The serialized handlers remain useful for batch size one. A multi-request
worker instead uses `pretrained.continuation.paged-runtime`: bounded concurrent
controller tasks enqueue work into one device-owning loop. That loop runs the
pure iteration policy, retains stable physical lanes, primes only refills and
prompt-token rows, and submits sparse active subsets to the fixed Raster graph.
Decode receives priority while a multi-lane worker reserves one lane-token for
waiting prefill work. A single-lane worker finishes active decode before
admitting another prompt, avoiding implicit time-slicing overhead.

Worker assembly keeps ownership explicit:

```clojure
(def runtime (paged-runtime/open-runtime decoder))
(def endpoint
  (controller-kabel/open-worker-endpoint
   (:pool decoder) worker-options
   (merge (paged-runtime/controller-submission runtime 64)
          {:handlers (paged/batched-handlers runtime cache decoder)
           :measurements measured-worker-state})))
```

The assignment-aware cancellation callback marks runtime jobs immediately but
wakes handler tasks only after the current append or restore boundary. This
keeps reservation release from racing in-flight GPU work. Close `endpoint`
before `runtime` so controller tasks quiesce first.

For batched handlers, restore no longer runs on the decoder-owning loop. A
bounded Raster `ContentProvider` localizes a Hasch-addressed chunk into the
worker's Konserve filestore, opens its Boring payload as a scoped segment, and
transfers that lease to a Raster upload event. The handler polls storage and GPU
events; short session locks do not prevent unrelated lanes from submitting
compute. Only event completion activates the route for prefill/decode.

Cancellation fences the assignment immediately but retains unused reserved
pages until any accepted local operation quiesces; terminal completion then
releases them. Pages already claimed by the continuation remain resident. A
shared partial tail reserves its possible copy-on-write page. This avoids both
use-after-release races and accepting a request that can restore successfully
but fail halfway through its declared decode budget.

`pretrained.continuation.controller.wire` is the narrow Kabel seam. It encodes
only offers, cancellations, offer results, and terminal results as ordinary EDN
maps with a `:type` field. Timers, GPU operations, reservation handles, and
tensor bytes are rejected as local-only. The same maps can be routed by Netz.
Worker observations use a separate versioned heartbeat message. The pure
`discovery` registry rejects delayed epochs/sequences without writing high-rate
heartbeats into Datahike.

`pretrained.continuation.controller.kabel` is the live interpreter for that
seam. Its router middleware associates each accepted heartbeat with the exact
connection that carried it, expires silent workers, and turns disconnects into
fenced `:worker/unavailable` events. Its worker middleware publishes page-pool
and queue observations, consumes directed offers/cancellations, and returns
acknowledgements/results. Unrelated messages pass through unchanged, so the
same peer can also carry Datahike/Konserve Sync and distributed-scope traffic.
Stale connections cannot remove or replace a newer worker route.

## Routing policy

Candidates are exact-prefix observations for one request. The initial score is
predicted time to first token:

```text
max(queue delay, lower-tier load) + GPU restore
  + uncached prompt tokens * measured prefill/token + first decode token
```

`controller.candidates` plans the request's exact chunk chain, fetches catalog
entries and all relevant replica facts in batch, and combines them with current
worker observations. It scores GPU, every usable RAM/SSD/object prefix boundary,
and recomputation. The lowest score wins, not necessarily the longest prefix.
Lower-tier alternatives remain available after a stale GPU decline. Capacity,
context limit, model availability, worker epoch, and exactness are hard
constraints. The worker repeats capacity admission against current page state
because a candidate can be stale by the time its offer arrives.

This policy is intentionally deterministic and inspectable. Datahike should
eventually record the candidate snapshot, selected alternative, predicted cost,
decline/retry reason, and measured outcome. High-rate queue and kernel samples
remain locally aggregated; the database stores decision-grade observations.

## Consumer and model interfaces

The first external surface should implement the small OpenAI-compatible subset
needed by common clients: model selection, chat/completion input, maximum new
tokens, cancellation, and streamed token deltas. The ingress adapter owns chat
templating and tokenization, then submits the protocol request. The controller
does not depend on HTTP or OpenAI JSON.

Hugging Face compatibility is a different boundary: pretrained-rstr loads model
artifacts, tokenizer/config metadata, and architecture adapters. It does not
need to reproduce every Transformers serving API. A later TGI-compatible
adapter is useful only if real users require it.

The current worker protocol returns a terminal token vector. Streaming deltas
and usage accounting should be added as fenced, nonterminal worker events before
an OpenAI server is advertised as complete.

## Next executable slice

The remaining Gemma path is concrete:

1. measure retained asynchronous checkpoint downloads under live decode and
   tune their transfer-stream priority and pinned-memory budget;
2. run cold, local-SSD, resident-prefix, partial-prefix, cancellation, and
   worker-restart cases with one small Gemma model;
3. report TTFT, inter-token latency, page occupancy, bytes by tier, recomputed
   tokens, eviction reasons, and inference/checkpoint overlap.

The live model-free demo now executes the complete observation, selection,
offer, acknowledgement, and result path across two local Kabel WebSocket
connections. The implementation does not yet claim a production deployment,
token streaming, multi-process Gemma execution, or LMCache-beating throughput.
Localize/upload and device/host checkpoint capture are now scheduled outside the
decoder loop, and checkpoint chunks have byte-aware host-staging admission.
Physical transfer/compute overlap is not yet guaranteed by Raster's current
OpenCL or Level Zero mappings. Cancellation preserves the current restore event
boundary before releasing a partial route. Fragmented staging pipelines, an
independent copy path, pinned host-memory pooling, and real multi-process Gemma
measurements remain required before making throughput claims.
