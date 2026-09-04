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

## HTTP adapter acceptance criteria

The optional HTTP-kit adapter is the next slice. It must:

1. expose chat completions and the configured model ids;
2. emit `text/event-stream` frames ending in `data: [DONE]`;
3. aggregate the same terminal tokens for non-streamed responses;
4. cancel the controller request when the client disconnects;
5. bound each client's output queue so a slow socket cannot stall the
   device-owning decode loop;
6. return OpenAI-shaped JSON errors with appropriate HTTP status codes;
7. pass an end-to-end test using a standard OpenAI client with only `base_url`
   changed.

Authentication, quotas, TLS termination, and production rate limiting belong
at the deployment boundary and are not part of the first embedded server.
