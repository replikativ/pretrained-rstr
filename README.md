# pretrained-rstr

[![Clojars Project](https://img.shields.io/clojars/v/org.replikativ/pretrained-rstr.svg)](https://clojars.org/org.replikativ/pretrained-rstr)
[![CircleCI](https://circleci.com/gh/replikativ/pretrained-rstr.svg?style=shield)](https://circleci.com/gh/replikativ/pretrained-rstr)
[![Slack](https://img.shields.io/badge/slack-join_chat-brightgreen.svg)](https://clojurians.slack.com/archives/C09622F337D)

> ⚠️ **Experimental**: pretrained-rstr is under active development. APIs may change before 1.0. Feedback welcome!

Run pretrained HuggingFace models — **text embeddings, speech-to-text, and decoder
LLMs** — natively on the JVM, on the [raster](https://github.com/replikativ/raster)
typed-dispatch compiler. No Python, no ONNX runtime: weights load from safetensors,
quantize to int8/int4 streams, and run on raster's CPU int8-MAC kernels and
Intel-GPU (Level Zero/OpenCL) programs.

```clojure
(require '[pretrained.embed :as emb]
         '[pretrained.asr :as asr]
         '[pretrained.lm :as lm])

;; every modality is the same shape: (load-X :key [dir] [opts]) then a task verb.
;; a registry key auto-downloads weights from HF on first use; an explicit local dir
;; skips the download (bring your own weights for a known model).

;; embeddings
(def e (emb/load-embedder :qwen3-embedding-0.6b))
(emb/embed-texts e ["Datahike is a durable Datalog database."])
;; => {:data float[n*1024] :n 1 :dim 1024}   (L2-normalized rows)

;; speech-to-text — any audio format (wav pure-JVM; mp3/ogg-opus/m4a via ffmpeg)
(def m (asr/load-asr :moonshine-streaming-medium))
(asr/transcribe m "voice-note.oga")
;; => "And so my fellow Americans, ask not what your country can do for you, ..."
(asr/transcribe m "talk.wav" {:timestamps? true})
;; => {:text "..." :words [{:word "And" :start 0.0 :end 0.02} ...]}

;; decoder LLMs
(def g (lm/load-lm :gemma-3-270m-it))                ;; or (lm/load-lm :gemma-3-270m-it {:gpu? true})
(lm/generate-text g "The capital of France is" 20)
;; => " Paris. ..."
```

## Models

| Registry key | Task | Size | Quality (validated) |
|---|---|---|---|
| `:qwen3-embedding-0.6b` (+`-gpu`) | embeddings, last-token | 0.6B Q8 | cos 0.999 vs torch f32 |
| `:embeddinggemma-300m` | embeddings, bidirectional + mean pool | 300M Q8 (GPU) | cos 0.99 vs torch; 768d matryoshka |
| `:all-minilm-l6-v2`, `:bge-small-en-v1.5` | embeddings (BERT tier) | 23–33M f32 | parity with sentence-transformers |
| `:moonshine-streaming-medium` | English ASR, **true streaming** | 245M | **WER 1.62% == HF torch** (LibriSpeech-100); word timestamps |
| `:qwen3-asr-0.6b` / `-1.7b` | multilingual ASR (52 languages) | 0.6/1.7B | transcript char-identical to torch gold |
| `:gemma-3-270m-it` / `:gemma-3-1b-it` | decoder LLM | ≤1B Q4/Q8 | token-exact GPU decode vs oracle; CPU ≈ llama.cpp speed |
| `:qwen3-0.6b` / `:qwen3-1.7b`, `:smollm2-135m-instruct` / `:smollm2-360m-instruct` | decoder LLMs | ≤1.7B Q4/Q8 | same descriptor-driven engine (shared attention/norm stack) |

Embeddings feed directly into [proximum](https://github.com/replikativ/proximum)
(`emb/rows` → HNSW vector index) and [umap-rstr](https://github.com/replikativ/umap-rstr)
(`emb/flat-doubles` → 2-D layouts). BERT-family sentence encoders (MiniLM/bge, mean-pool)
run self-contained in `pretrained.arch.bert` — no extra dependency; `:engine :encoder`
registry entries route there automatically.

## How it works

A model architecture is a *descriptor* — a role→tensor-name map plus ~10 flags (norm
type/gain, rope variants, GQA, qk-norm, sliding windows, sandwich norms, MoE routing)
— interpreted by one generic engine over raster's compilable `deftm` blocks. Adding a
standard decoder-LM is a descriptor, not engine code.

- `pretrained.embed` / `pretrained.asr` / `pretrained.lm` — task-level APIs, all the same
  shape: a curated registry + HF auto-download + `load-X`/task-verb (CPU or `{:gpu? true}`)
- `pretrained.decoder` — the descriptor-driven decode engine (`load-hf`, `decode-step`,
  `generate-cached`); GPU-resident decode/prefill in `pretrained.decoder-gpu`
- `pretrained.loader` — the low-level generic loader: architecture registry, tokenizer
  auto-detection, `from-pretrained` (dispatch any HF dir on `config.json` model_type — the
  advanced, unvalidated path behind the curated `pretrained.lm` registry)
- `pretrained.arch.*` — decoder descriptors as pure data: `gemma3`, `llama`, `qwen3`,
  `qwen3-moe`, `embedding-gemma`, plus the self-contained BERT encoder `bert`;
  `pretrained.asr.*` — moonshine, qwen3-asr
- `pretrained.tokenizer.{sp,bpe,wordpiece}` — HF `tokenizer.json` tokenizers
- `pretrained.hub` — sha-pinned downloads with resume + sha256 into
  `~/.cache/raster/models` (`HF_TOKEN` honored; every `load-*` also accepts a local dir)
- `pretrained.safetensors` / `pretrained.audio` — format frontends (bf16/f16 fast paths;
  WAV pure-JVM, other audio via ffmpeg)

**Quantized execution:** linear weights repack into int8/int4 streams executed by
raster's spin-pool int8-MAC kernel (CPU) or dp4a kernels (Intel GPU). Q8_0 is measured
lossless for embeddings; decode uses Q4. Quantized streams are disk-cached next to the
weights — the first load quantizes once (~30s for 0.6B), warm loads take ~5s.

## Validation methodology

Every port is validated against its reference implementation before it ships:
layer-by-layer activation comparison vs HF transformers golds (typical agreement
~1e-6 relative in f32), then end-to-end anchors — token-exact decode, character-exact
transcripts, cos ≥ 0.999 embeddings vs torch f32. The anchors are repeatable tests:

```
clojure -M:test        # fast, model-free unit tests
# with local weights (see test/pretrained/anchors_test.clj):
clojure -A:dev:test:valhalla -M -e "(require 'clojure.test 'pretrained.anchors-test) \
  (clojure.test/run-tests 'pretrained.anchors-test)"
```

## Requirements

- JDK 21+ (Panama FFI); the raster Valhalla toolchain for full performance
  (see raster's README for the JVM flags)
- OpenBLAS for the f32 GEMM paths
- ffmpeg (optional, for non-WAV audio)
- Intel GPU (optional): Level Zero or OpenCL runtime for the GPU decode/prefill paths

## License

Copyright © 2026 Christian Weilbach

pretrained-rstr is [MIT licensed](LICENSE). Model **weights** carry their own licenses
(all registry models are Apache-2.0/MIT) — you are responsible for complying with each
model's terms.
