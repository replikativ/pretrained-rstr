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

Integration tests exercise model listing, streamed and non-streamed chat,
usage, errors, and disconnect cancellation over a real TCP listener. A standard
OpenAI SDK smoke test with only `base_url` changed is the remaining compatibility
gate before calling the endpoint complete.

Authentication, quotas, TLS termination, and production rate limiting belong
at the deployment boundary and are not part of the first embedded server.
