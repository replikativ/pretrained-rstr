# OpenAI-compatible serving boundary

The first public serving surface is a deliberately small, tested subset of
`POST /v1/chat/completions`. Compatibility is an ingress/egress concern; HTTP,
JSON, chat templates, and tokenization do not enter the cluster router or GPU
worker state machines.

```text
OpenAI JSON -> chat template/tokenizer -> generation request -> cluster router
     ^                                                        |
     +---- JSON response or SSE chunks <- fenced deliveries <-+
```

## Initial contract

Accepted request fields are:

- `model`;
- nonempty text `messages` with `system`, `developer`, `user`, or `assistant`
  roles;
- `max_completion_tokens`, with `max_tokens` accepted as an alias;
- `stream` and `stream_options.include_usage`;
- `temperature`, `top_p`, `stop`, and `seed`.

Known fields with unsupported values produce an OpenAI-shaped
`invalid_request_error`. Additional fields are currently ignored and must not
be advertised as implemented. Tool calls, structured outputs, multimodal
content, log probabilities, and the Responses API are later compatibility
slices.

`pretrained.openai/normalize-chat-request` accepts injected model resolution and
chat tokenization, returning the ordinary continuation `generation-request`.
The controller therefore has no dependency on OpenAI JSON.

## Streaming safety

Each generated token carries its assignment id and a zero-based index through
the worker wire protocol. Both worker and router accept only the next index for
the active assignment. This fences duplicate, stale, reordered, and late
messages.

Before any token reaches the consumer, offer failure or worker loss can select a
new candidate. After the first delta, transparent retry is unsafe because a new
worker could duplicate or diverge from already visible output. The current
contract closes the stream with an error. A future retryable stream must first
publish or transfer an exact continuation at the last delivered token boundary.

## HTTP adapter

`pretrained.openai.server` is available with the `:openai-server` alias and the
Replikativ HTTP-kit distribution. It:

1. exposes chat completions and configured model ids;
2. emits `text/event-stream` frames ending in `data: [DONE]`;
3. aggregates terminal tokens for non-streamed responses;
4. cancels the controller request when the client disconnects;
5. bounds each client's application queue and observes HTTP-kit's queued-byte
   high/low watermarks, keeping a slow socket off controller and decoder loops;
6. configures a hard per-connection queued-byte ceiling;
7. returns OpenAI-shaped JSON errors with appropriate HTTP status codes.

Terminal usage reports the actual reusable KV prefix as
`usage.prompt_tokens_details.cached_tokens`. The worker records the count
returned by restore, rather than the router's potentially stale cache estimate,
and carries it through the assignment-fenced terminal result.

Integration tests exercise model listing, streamed and non-streamed chat,
usage, errors, and disconnect cancellation over a real TCP listener. The same
surface has been exercised with the Python `openai` 2.21.0 client for model
listing, non-streamed completion, streamed deltas, and streamed usage with only
`base_url` redirected:

```python
from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:8080/v1", api_key="local")
result = client.chat.completions.create(
    model="gemma-3-270m-it",
    messages=[{"role": "user", "content": "Explain KV caching briefly."}],
    stream=True,
    stream_options={"include_usage": True},
)

for chunk in result:
    if chunk.choices:
        print(chunk.choices[0].delta.content or "", end="", flush=True)
```

The placeholder API key satisfies the client constructor; the embedded server
does not currently authenticate it.

## Cluster composition

`pretrained.openai.cluster/open-server` owns the lifecycle wiring between HTTP
ingress and a Datahike-backed Kabel router. Worker sockets attach through its
`router-middleware`; only control messages and token results cross Kabel, while
KV chunks remain on the Konserve placement path. The model-free live showcase
runs two actual worker WebSockets behind one HTTP/SSE listener:

```clojure
;; clojure -M:distributed-demo
(require '[pretrained.continuation-kabel-demo :as demo])
(demo/run-openai-live-simulation)
;; => {:http-status 200, :selected-worker :fast-gpu,
;;     :text "<101><102>", :cached-token-count 0, ...}
```

Its opt-in regression test is:

```sh
clojure -M:test:distributed-demo:distributed-test -d test-distributed
```

For an opt-in topology in which both Kabel workers own independent real Raster
paged decoders, use the resource-gated smoke below. It defaults to the local
`gemma-3-270m-it` checkpoint and proves that a continued OpenAI request reports
positive resident cached-token usage:

```sh
clojure -M:valhalla:distributed-demo:real-cluster-test -e \
  "(require '[pretrained.real-openai-cluster-demo :as demo]) (demo/run!)"
```

Inspect `(demo/preflight)` first. The smoke refuses to load weights under
memory, swap, or host-load pressure. `:force? true` bypasses that guard and
should only be used after checking competing host and integrated-GPU workloads.

Authentication, quotas, TLS termination, and production rate limiting belong
at the deployment boundary and are not part of the first embedded server.
