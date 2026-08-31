# Numerical memory beyond LLM inference

The continuation cache is a working instance of a broader programming model:
large numerical state is immutable and content-addressed at durable boundaries,
small semantic facts describe its lineage and placement, and compiled programs
operate on worker-local resident views.

This repository proves that model for LLM attention state. It does not yet ship
a weather model, PDE solver, MPI runtime, or direct cross-node GPU transport.
Those are a research and integration direction.

## A three-plane architecture

| Plane | Current stack | Responsibility |
| --- | --- | --- |
| semantic/control | Datahike, optionally Yggdrasil/Simmis | identity, provenance, branches, dependencies, policy, observations, accepted results |
| durable numerical data | Konserve, local files, S3-compatible stores | immutable tensor/field chunks, content addressing, differential replication |
| resident compute | Raster | typed numerical programs, device allocation, schedules, transfers, CPU/GPU execution |

This separation matters at scale. Datoms are well suited to questions such as
“which model and boundary conditions produced this field?”, “which worker has a
compatible replica?”, and “which scenario was accepted?”. They are the wrong
representation for billions of adjacent floating-point cells. Bulk arrays stay
contiguous and mmap/range-transfer friendly; Datahike stores content references
and queryable metadata.

## From a KV cache to a simulation checkpoint

The mapping to a time-stepped scientific model is direct enough to guide an
experiment:

| LLM inference | Time-stepped simulation |
| --- | --- |
| model fingerprint | solver, mesh, equations, parameters, precision, compiler fingerprint |
| token-prefix hash | causal checkpoint/timestep identity |
| K/V slabs by layer | field tiles by variable and level |
| pending token | next timestep or integration stage |
| prefix fork | scenario/ensemble branch |
| paged attention working set | resident stencil/domain working set |
| restore vs recompute policy | checkpoint reload vs replay from an earlier step |

A weather-oriented prototype could chunk pressure, temperature, humidity, and
wind fields by spatial tile and vertical level. Datahike would index run lineage,
initial conditions, observation-assimilation inputs, solver version, timestep,
and the content references for each field tile. Raster programs would consume
resident tiles and produce the next immutable boundary. Branching a forecast
scenario would share all unchanged checkpoint objects until the paths diverge.

That produces useful properties before attempting a full atmospheric model:

- reproducible runs with explicit numerical and compiler compatibility;
- cheap ensemble/scenario forks through structural sharing;
- deduplicated checkpoints across common histories;
- local or object-store-backed restore without routing bulk arrays through the
  database writer;
- queryable provenance from assumptions and observations to generated fields;
- an adoption boundary where Simmis can distinguish explored futures from an
  accepted organizational plan.

## What would need to change

The present chunk format is attention-state specific even though its storage
principles are general. A simulation implementation should first extract a
versioned numerical-state manifest with named slabs, shapes, dtypes, coordinate
domains, and causal parents. Compatibility must include the solver and numerical
scheme, not only tensor shape.

The resident runtime would then need domain-specific execution mechanisms:

- stencil and sparse/structured-grid schedules;
- halo exchange and boundary-condition semantics;
- multi-device partitioning and topology-aware placement;
- asynchronous checkpoint streams that do not stall solver kernels;
- possibly MPI, UCX, GPUDirect/RDMA, or vendor collectives for bulk peer
  transfers.

Those transports belong below the semantic state model. Datahike should record
topology, replicas, decisions, and outcomes; it should not mediate every halo
exchange or kernel event. Raster should expose backend-neutral transfer and
collective contracts while specialized backends choose the actual transport.

## A staged research path

1. Generalize the manifest and content identity around a small CPU field solver.
2. Demonstrate fork, exact restore, recompute comparison, and Datahike provenance
   on a two-dimensional stencil or shallow-water model.
3. Move tiles into Raster resident buffers and overlap compute with durable
   checkpointing on one GPU.
4. Add ensemble scheduling across several local devices or workers, retaining
   the same control/data-plane split.
5. Introduce halo exchange and direct-memory transports only after profiling
   shows where copies and coordination dominate.

This progression connects the present inference system to the Simmis vision of
comparing possible futures without claiming that branching alone supplies a
causal or scientifically valid model. The equations, assumptions, calibration,
and validation remain the substance of the simulation; the stack supplies a
durable and inspectable way to run, fork, compare, and retain it.
