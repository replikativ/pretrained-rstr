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

Datahike: model, lineage, placement, observations, recorded decisions
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
5. invokes injected restore, prefill, and decode handlers on one serialized
   execution queue.

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

## Routing policy

Candidates are exact-prefix observations for one request. The initial score is
predicted time to first token:

```text
max(queue delay, lower-tier load) + GPU restore
  + uncached prompt tokens * measured prefill/token + first decode token
```

The lowest score wins, not necessarily the longest prefix. Capacity, context
limit, model availability, worker epoch, and exactness are hard constraints.
The worker repeats capacity admission against current page state because a
candidate can be stale by the time its offer arrives.

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

1. derive request-specific candidates from the local Datahike replica and
   worker observations;
2. connect the local restore handler to `restore-paged-prefix!`, passing the
   accepted capacity reservation;
3. connect suffix work and decode to the existing paged scheduler/lane refill;
4. checkpoint completed immutable ranges asynchronously through the tiered
   Konserve store and publish catalog facts only after durability receipts;
5. run cold, local-SSD, resident-prefix, partial-prefix, cancellation, and
   worker-restart cases with one small Gemma model;
6. report TTFT, inter-token latency, page occupancy, bytes by tier, recomputed
   tokens, eviction reasons, and inference/checkpoint overlap.

The implementation does not yet claim a production Kabel deployment, token
streaming, multi-process Gemma execution, or LMCache-beating throughput. Those
claims require the handler integration and measurements above. The controller
and simulator now make those experiments bounded and reproducible.
