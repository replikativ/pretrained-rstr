# Local and private model roadmap

`pretrained-rstr` is optimized for models that an organization can run and
adapt on its own hardware. The primary use case is private inference and
fine-tuning of small models, with durable attention state and training artifacts
coordinated through Datahike. Large checkpoints are useful compatibility probes,
but they are not the center of the project.

## Product boundary

The default path should remain useful on an integrated GPU or a modest discrete
GPU:

- load a curated Hugging Face checkpoint without a Python runtime;
- run a numerically anchored CPU or Raster device implementation;
- expose the model through the OpenAI-compatible HTTP/SSE boundary;
- reuse, branch, persist, and relocate exact KV prefixes;
- fine-tune locally, initially through parameter-efficient adapters; and
- keep prompts, continuations, adapters, and provenance under the operator's
  control.

The distributed programming model is not restricted to one machine. A private
deployment can place workers near data, advertise exact resident prefixes, and
move immutable numerical state through the Datahike/Konserve stack without
routing tensor payloads through the HTTP control plane.

## Uniform model contract

Architecture support should converge on one descriptor-driven contract instead
of accumulating model-specific serving paths. A family implementation owns:

1. Hugging Face configuration and tensor-name mapping;
2. tokenizer and chat-template behavior;
3. attention, normalization, positional encoding, and feed-forward descriptors;
4. CPU reference execution and Raster device lowering;
5. supported weight and activation quantization variants;
6. attention-state layout and compatibility fingerprint inputs; and
7. optional trainable/adaptable parameter groups.

Task APIs (`pretrained.lm`, embedding, and ASR) remain curated entry points.
`pretrained.loader/from-pretrained` remains the bring-your-own-checkpoint escape
hatch, but recognizing an architecture is not itself a support claim.

## Evidence tiers

Public documentation and the model registry should distinguish these tiers:

| Tier | Required evidence |
| --- | --- |
| Recognized | Configuration dispatches, tensors and tokenizer load, shapes validate. |
| Runnable | A bounded CPU or device generation smoke test completes with finite, non-degenerate output. |
| Anchored | Layer or token output is compared with an independent reference on a pinned checkpoint. |
| Continuation-ready | Paged decode, copy-on-write fork, restore, and uninterrupted-vs-resumed token equality pass. |
| Serving-ready | OpenAI non-streaming/streaming behavior, cancellation, backpressure, and multi-worker routing pass. |
| Adaptation-ready | A pinned adapter/fine-tuning fixture demonstrates loss movement, save/load, provenance, and unchanged base weights. |

The registry should eventually expose this evidence as data so model listing and
documentation can be generated rather than hand-maintained.

## Model envelope

- **Primary:** roughly 100M-1B local instruction, embedding, reranking, and
  speech models; Q4/Q8 inference; LoRA or similarly bounded fine-tuning.
- **Routine compatibility:** models up to roughly 3B when hardware and memory
  permit. These may be slow, but loading and bounded inference should be tested.
- **Optional stress probes:** quantized 7B-class checkpoints. They are useful for
  finding address-space, quantization, compiler, and paging defects, but are not
  required for ordinary CI or release readiness.

Limits are capability- and hardware-dependent rather than promises that every
checkpoint of a given parameter count will fit.

## Near-term sequence

1. Publish a machine-readable model capability/evidence registry for the
   existing Gemma, Qwen, Llama/SmolLM, embedding, and ASR families.
2. Make tokenizer chat templates and stop-token policy part of the uniform model
   contract used by the OpenAI adapter.
3. Add resource-gated real-model tests for two-worker routing, resident-prefix
   reuse, durable restore, and restart recovery.
4. Normalize Q4/Q8 selection and record the execution variant in attention-state
   compatibility fingerprints.
5. Add a small, reproducible LoRA training fixture whose adapter checkpoint and
   optimizer provenance are stored as immutable numerical artifacts.
6. Add optional slow lanes for a 3B checkpoint and a quantized 7B checkpoint;
   report measured memory and latency without making them release gates.

This sequence keeps privacy and useful local adaptation ahead of broad model
count, while the shared descriptor and evidence model steadily expands reach.
