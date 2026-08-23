# Handoff: durable KV continuations on raster — instructions for the implementing agent

Written 2026-08-22 after landing the raster side (PRs #93, #95, #96). Read this whole file before
touching code. Everything stated as fact here was verified, not assumed; where something is
unverified it says so.

## 0. Where things stand

**Raster is ready.** `org.replikativ/raster` main now has the two things this work needs:

- `raster.gpu.core/upload-range!` / `download-range!` — move a sub-range of a session buffer.
- `raster.gpu.core/upload-ranges!` / `download-ranges!` — the batched form, one call for all
  layers, **all-or-nothing on validation**.

**Pin `org.replikativ/raster` to `0.2.320`.** That is the Clojars release of #96 (tag `v0.2.320`,
confirmed cut 2026-08-22). The `:dev` alias (`:local/root "../raster"`) is the same code and is how
everything below was tested.

**pretrained-rstr currently pins `0.2.287` — 32 commits behind.** The upgrade is safe; evidence in
§2. The CPU continuation contract (§4, step 1) depends on **nothing** in raster and can start
before the bump lands.

## 1. Branch setup

Base on `main`, **not** `feat/deepseek-ocr-decoder` (17 commits of OCR/R-SWA work; R-SWA's
pinned-prefix-plus-ring semantics complicate the first continuation contract — useful later).

A worktree on main already exists: `../pretrained-rstr-main` (at `e24f9fc`). Use it:

```
cd ../pretrained-rstr-main
git checkout -b feat/durable-kv-continuations
```

## 2. The raster bump — do this first, it is mechanical

Edit `deps.edn`: `org.replikativ/raster {:mvn/version "0.2.287"}` → `"0.2.320"`.

**Evidence the bump is safe** (all run against raster main via `:dev`, on this machine, 2026-08-22):

| check | result |
|---|---|
| every raster symbol this repo calls (15) still resolves | 15/15 |
| model-free suite `clojure -M:test` | 8 tests / 39 assertions, green |
| GPU decoder oracle (gemma-270m shapes on the real Arc, through the CHANGED contraction routing) | 2 tests / 9 assertions, green |
| `pretrained.arch.bert` loads and monomorphizes | yes |

One thing chased specifically: raster commit `917c1250` types `(All [T])` kernel scalars as `:- T`.
This repo has 3 `(All [T])` kernels in `arch/bert.clj` passing `eps :- Double`. **Not a problem**:
raster's own `nn/layer-norm` still declares `eps :- Double`, so bert matches raster's contract.

**Re-run exactly these after bumping**, as the regression gate:

```
clojure -M:test                                  # CI parity (model-free)
clojure -M:dev -e "(require 'pretrained.decoder-gpu-oracle-test)
  (let [r (clojure.test/run-tests 'pretrained.decoder-gpu-oracle-test)]
    (System/exit (if (zero? (+ (:fail r) (:error r))) 0 1)))"
```

⚠ **The GPU oracle must be invoked via `clojure.test` directly, as above.** The `:test` alias
bakes in `-e :anchors`, and the oracle is `^:anchors`-tagged, so `clojure -M:test -n
pretrained.decoder-gpu-oracle-test` **silently runs zero tests and exits 0**. This was hit twice.
Confirm with the `Testing pretrained.decoder-gpu-oracle-test` line in the output.

## 3. The raster API you will use

Session buffers are allocated `alloc-shared` (host-coherent). The KV cache in `decoder_gpu.clj`
(`kv-specs`, ~line 316) is per-layer `kc<l>` / `vc<l>` buffers, each `[maxpos][kvrow]`
**position-major**, `kvrow = n-kv × head-dim`. So a prefix of `t` tokens in one layer is ONE
contiguous range of `(* t kvrow)` elements starting at element 0.

```clojure
(require '[raster.gpu.core :as g])

;; export a t-token prefix of every layer, in one call
(g/download-ranges! sess
  (for [l (range n-layers) kind ["kc" "vc"]]
    [(keyword (str kind l)) (dst-for l kind) {:src-element 0 :elements (* t kvrow)}]))

;; restore it
(g/upload-ranges! sess
  (for [l (range n-layers) kind ["kc" "vc"]]
    [(keyword (str kind l)) (src-for l kind) {:dst-element 0 :elements (* t kvrow)}]))
```

Semantics you can rely on (each is a device test in raster, mutation-checked):

- `src`/`dst` is a JVM primitive array **or** a `java.lang.foreign.MemorySegment`. A segment —
  e.g. a Boring mmap — is copied **directly** into the shared allocation, no JVM array in between.
  Measured: 722 B of heap per segment upload vs a 131 KB payload.
- Offsets (`:src-element`, `:dst-element`, default 0) and `:elements` are in **elements of the
  buffer's dtype**. You never compute bytes.
- **Out-of-range is an error, never a clamp.** (`array->buffer!`, the whole-buffer path, clamps —
  that is fine for it and exactly wrong for a range.)
- **Nothing outside the range moves.** Tested: regions before and after an import are untouched.
- **Batch is all-or-nothing on validation**: every entry is bounds-checked before any is copied, so
  a bad spec in layer 30 cannot leave layers 1–29 written. A fault *during* execution is still
  partial — different class, not promised.
- Results come back **in entry order**.
- Transfers are **synchronous**. Async/event-based overlap of restore with decode is not there yet
  — it is an additive follow-up in raster, not something to work around here.

What is deliberately **not** provided: foreign `DeviceArray` rebinding, exposing private buffers, a
paged/slot-map gather. The first two are north-star §2.3 work; the third is only needed once the
cache is block-paged (see §6). Do not build around their absence.

## 4. Build order — vertical commits, each with a test against uninterrupted generation

1. **CPU continuation state + contract.** Processed-token count, pending token/row, model
   fingerprint, cache layout descriptor, K/V contents. `decoder.clj:~451` currently allocates and
   discards its cache; `decoder_gpu.clj:~628` returns only session/model metadata — there is no
   continuation abstraction to extend, you are creating it. **Needs nothing from raster.**
2. **Token-exact export/import tests.** Generate N tokens uninterrupted; generate k, export,
   import into a fresh state, generate N−k; assert identical token ids. This is the correctness
   anchor for everything after. Use gemma-3-270m (18 layers, 1 kv-head, head-dim 256 → ~36 KiB
   per token FP32; 9 MiB / 256 tokens). `lm.clj:~16` has the strongest existing anchors.
3. **GPU export/import** via `download-ranges!`/`upload-ranges!`. Same token-exact test, GPU
   path. This is the step that needs the raster bump.
4. **Boring physical storage + mmap restoration.** One immutable blob per continuation: small
   indexed CBOR manifest + typed K/V slabs per layer. Restore = mmap → `MemorySegment` →
   `upload-ranges!` directly (that is what the segment path is for).
5. **Datahike catalog.** Identity, reachability, layout, policy, history in datoms; the Boring
   content id as `:db.type/store-ref` (`../datahike/doc/store-refs.md`). Tensor bytes never in
   datoms. Batch access counters — do not transact per token.
6. **Local cache manager**: `checkpoint!`, `lookup`, `ensure-resident!`, `lease!`, `release!`,
   `evict!`. The manager runs GPU effects; Datahike records facts; a reconciler reads facts and owns
   live leases and transfers.
7. **REPL showcase + benchmark.**

Suggested catalog shape (from the design discussion; adjust as the contract firms up):

```clojure
{:kv/id                continuation-id
 :kv/model-fingerprint model-hash
 :kv/prefix-hash       token-prefix-hash     ; content hash of the token prefix — see §6
 :kv/token-count       256
 :kv/dtype             :float32
 :kv/layout            :layer-kv-token-head-d
 :kv/bytes             9437184
 :kv/blob              boring-content-id      ; :db.type/store-ref
 :kv/created-at        instant}
```

## 5. Benchmark honesty — read this before producing a number

Compare cold prefill / RAM restore / mmap restore / uninterrupted resident continuation. **Do not
benchmark against the 4.9 s batched-prefill figure.** That is the opt-in path; the default is
per-token priming at ~1.2 s, and `asr/qwen3_asr.clj:~561` documents why (420 kernels,
occupancy-bound). Against 4.9 s a cache would look ~4× better than it is. **Baseline is ~1.2 s.**

Also: raster's `launch-2d!` has a ~0.8 ms fixed cost per launch. Anything under ~10 ms of device
work is overhead-dominated — scale the measurement until overhead is <5% or the number lies.

## 6. LMCache comparison — what transfers over, what does not

`../LMCache` was studied. Its GPU connector has **exactly** our shape: `to_gpu(obj, start, end)` /
`from_gpu(...)` plus `batched_*`. Three differences, all deliberate:

- **No gather kernel.** vLLM's KV is block-paged, so LMCache runs a CUDA gather
  (`csrc/cuda/mem_kernels.cu`) from a `slot_mapping`. Ours is position-major, so the range copy IS
  the gather — an advantage. This holds until the cache is paged for block-level prefix reuse /
  dedup across continuations; that is the point a slot-map gather becomes necessary. Plan for it,
  do not pre-build it.
- **Per-layer buffers, not one `[2, L, T, D]` blob.** LMCache's default `MemoryObj` is one blob
  for all layers; it also has a `Layerwise` connector variant. Ours is the layerwise shape. The
  batched call is the mitigation for the N-calls cost.
- **Synchronous.** LMCache copies on dedicated streams with events. Ours blocks. Fine for the
  prototype; needed once restore overlaps decode.

The chunk contract worth copying: LMCache's `chunk_size` is 256 tokens, retrieve is
**chunk-granular with overlap** (it will re-write tokens vLLM already has because the data is
identical), and it returns a per-token mask, not a count. That validates one-blob-first, then
immutable 64/128-token blocks — and it says the block boundary must be a **content hash of the
token prefix up to that block**, which is what `:kv/prefix-hash` is.

## 7. Deferred on purpose

- KV quantization — until the FP32 round trip is token-exact. FP16 is the first experiment; Q4/Q8
  KV must be justified by quality + transfer-vs-dequant measurement.
- Block granularity / longest-prefix reuse / dedup — after the single-blob prototype.
- Async transfers, foreign-buffer rebinding — raster-side follow-ups.

## 8. Rules of the road (repo conventions that apply here)

- Everything goes through a PR; do not push to main.
- `deftm`/`ftm` for numerical code; `raster.numeric` ops, `raster.arrays/aget`; integer index
  arithmetic stays `clojure.core` (see `../raster/CLAUDE.md`).
- Tests that depend on state outside the test (disk caches, device timing bars) are the flake class
  raster is currently cleaning up — write the token-exact tests against an in-memory reference, and
  give Boring/Datahike tests a temp directory fixture.

## 9. UPDATE 2026-08-22 — the `bind-decode!` census failure you reported is FIXED, but not in 0.2.320

Your `KV_CONTINUATION_GPU_BIND_BUG.md` was right: the census was doing its job. Root cause was
`par/dp4a` being a plain `defn` (no return type), and fixing that exposed four further defects in
the GPU/wasm intrinsic lowering (raster PR #97, merged). `bind-decode!` on gemma-3-270m now returns
`:BIND-OK` on the Arc — verified against the exact repro in your report.

**It is released as `org.replikativ/raster 0.2.321`** (tag `v0.2.321`, cut 2026-08-22). Bump the
pin from `0.2.320` → `0.2.321` and re-run your gate:

    clojure -M:dev -e "(require 'pretrained.loader 'pretrained.decoder-gpu)
      (pretrained.decoder-gpu/bind-decode! (pretrained.loader/from-pretrained
        (str (System/getProperty \"user.home\") \"/Development/models/gemma-3-270m-it\")) :maxpos 64)"

Expected: a resident decoder state, no census throw. (0.2.320 still fails exactly as your report
describes — the fix is one release later.)

## 10. UPDATE 2026-08-23 — chunked local tier implemented

The branch now has both formats. Whole-prefix `.rstrkv` snapshots remain the archival fallback;
the reuse path stores immutable processed-token chunks (256 tokens by default) in a local
Konserve filestore using Boring.

The logical and physical flows are:

```
processed tokens -> parent-linked prefix hashes -> one batched Datahike lookup
GPU KV range      -> one contiguous FP32 array  -> one Konserve value per missing chunk
Datahike matches  -> Konserve mmap payload      -> Raster ranged upload at token offset
missing prompt suffix ---------------------------------------> ordinary decoder prefill
```

Important details:

- The pending token is outside the hash chain because its KV row does not exist yet.
- Each physical payload is ordered `K0..Kn,V0..Vn`, so Konserve 0.9.377 maps a chunk once and
  Raster slices that one segment for every layer.
- Datahike holds query/policy facts and a local `:kv/store-key`; local off-band chunks do not
  pretend to be Datahike `:db.type/store-ref` values. A database-owned blob adapter can be added
  later without changing the chain/query contract.
- Restore maps/uploads one chunk at a time, bounding mapped residency. Transfers are synchronous;
  this does not yet overlap SSD/device IO with decode.
- `checkpoint-gpu-chunks-async!` uses bounded low-priority workers. Submission does no cache IO,
  but the background device copy can still contend with inference on current backends.
- `restore-gpu-prefix` loads the longest root-contiguous match and computes only the prompt suffix.
- `dev/pretrained/kv_continuation_demo.clj` contains `run-gpu-prefix-reuse!` for an nREPL demo.

Model-free verification: `clojure -M:test` passes 31 tests / 142 assertions. The focused chunk
suite covers Datahike -> Konserve mmap -> Raster upload, including old-catalog schema migration.
