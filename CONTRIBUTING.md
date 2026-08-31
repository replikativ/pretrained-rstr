# Contributing to pretrained-rstr

pretrained-rstr is experimental systems software at the boundary between model
ports, numerical compilation, and durable inference state. Small, reviewable
changes with explicit validation evidence are easiest to land.

## Development setup

Use JDK 21 or newer and the Clojure CLI. OpenBLAS is required by the floating
point GEMM paths. The default dependency graph uses released artifacts:

```sh
clojure -P -M:test
clojure -M:test
```

The `:dev` alias replaces Raster with the sibling checkout at `../raster` and
adds the local development and test paths:

```sh
clojure -M:dev
```

Run this mode only when the sibling checkout is intentional. Its branch and
uncommitted state can differ from the released dependency used by CI.

## Validation levels

Choose the smallest level that establishes the claim being changed:

1. Model-free unit tests for formats, hashing, catalogs, scheduling, and pure
   execution plans: `clojure -M:test`.
2. Compiler/device tests for generated Raster programs. These require a
   compatible GPU runtime and should name the device/backend exercised.
3. Model anchors for architecture or numerical changes. Compare intermediate
   activations and final tokens against the upstream reference implementation.
4. Benchmarks for performance claims. Record hardware, backend, dtype, model,
   prompt/context length, warmup policy, and whether storage pages were cold or
   already in the process/page cache.

Do not weaken tolerances or update a reference output without explaining why
the old expectation was wrong. A faster result is not sufficient if parity is
lost.

## Repository boundaries

- Model discovery, tokenization, weight loading, architecture descriptors, and
  inference-state policy belong here.
- General tensor operations, compiler passes, schedules, and GPU runtime
  mechanisms belong in Raster.
- Queryable continuation identity, lineage, placement, and policy are Datahike
  data; bulk tensor payloads are not.
- Optional transports and object-store integrations stay out of the default
  dependency set when the embedded inference library does not require them.

## Pull requests

Describe the user-visible result, the validation performed, and any hardware or
model requirement reviewers need to reproduce it. Keep generated model weights,
quantized streams, credentials, local caches, and benchmark scratch data out of
Git.
