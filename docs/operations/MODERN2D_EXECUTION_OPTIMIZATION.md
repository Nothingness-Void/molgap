# PPGT / ESA execution optimization

This is an execution-only extension of the [configured workflow](MODERN2D_CONFIGURED_WORKFLOW.md).
It reuses FamilyKernel, Session, normalized-Gap updates, loaders, profiler and
atomic checkpoint owners. It does not introduce a trainer or change model
parameters, FP32/TF32-off policy, BS128, minibatch order, schedule or roles.

## Always-used shared changes

- `data/indexed_inputs.py`: `__getitems__` groups a requested minibatch by
  shard, checks each touched immutable file, separates rows directly without
  allocating PyG's shard-sized graph memo table, clones each occurrence and
  restores the exact sampler order. The two-handle mmap bound, chunk limits,
  source checks, role confinement and hash-on-file-identity semantics remain.
- `PreparedDevelopmentCache`: accepts exact non-contiguous source members.
  Indexed FamilyLoaders reuse complete ordered CPU development passes within
  the existing declared byte budget. Full role metadata is retained; overflow
  streams the entire role, interrupted passes never become a cache. No GPU
  corpus cache or changed validation frequency is introduced.
- `FamilyExecutionProfiler`: removes capture-only wrappers and hooks after
  capture, including fault paths. Update/loader observers remain for capture-off
  timing and cost accounting until normal context exit.
- `ModelSubstrate`: a newly attached addon container inherits its parent mode,
  including attachment to an already-eval model.

Checkpoint cadence, commit-last publication, independent integrity hashes and
output finite checks are deliberately unchanged. A mutable path is not enough
provenance to eliminate a checkpoint verification read safely.

## Explicit prepared execution candidate

`modern2d_execution_context.py` owns model-only CPU topology snapshots and the
call-scoped layout. It imports no training controller or registry. The factory
validates roles, categories, bounds, source IDs and graph boundaries before any
accelerator transfer. Snapshots retain neither labels nor RWSE; tensor identity,
version, shape, dtype and placement guards reject mutation or view replacement.

`modern2d_execution.prepared_forward(model, cpu_batch, device=...)` enters the
ordinary `model.__call__` hook frame; registered addons are not skipped. Scope
cleanup runs even when a forward fails. Historical `model.forward`, factory,
configuration, state keys and frozen recipes retain their original selection.

- PPGT: equal-node-count buckets remain within the original minibatch. Raw
  RRWP is computed with batched FP32 matrix powers and device-local per-graph
  asymmetric-union selection, avoiding a host boolean read and a series of
  tiny GEMMs per molecule. Edge multiplicity/orientation, isolates, positional
  channel order and output order are retained. Learned PE/bond/stem projections
  are recomputed every update. Stochastic presets keep graphwise RNG order.
- ESA: directed-bond/isolated-atom categorical tokens are packed on CPU and
  the four padded tensors are transferred once. Learned projections, M/S
  attention, padding masks and PMA equations remain the original dense path.
  This removes device-side split/reassemble/nonzero/max preparation; it does
  not claim sparse attention or remove inherent quadratic attention cost.

For a generated CPU fixture or an already admitted train-only native scratch
preflight, reuse the actual Session:

```python
from molgap.modern2d_execution import prepared_kernel_execution

with prepared_kernel_execution(kernel):
    # Optional existing FamilyExecutionProfiler goes inside this scope.
    session.run()
```

The context restores the kernel methods on exit. Real CUDA use is limited to
the admitted two-update preflight; a scientific training recipe is rejected.
The normal platform submission route does not silently enable this candidate.

## Qualification boundary

Synthetic CPU output/gradient/Adam/RNG/resume tests do not establish native
equivalence, accelerator speedup or model quality. In particular PPGT batched
GEMM need not be bitwise equivalent to graphwise GEMM; the prior graphwise vs
bucketed CPU discrepancy is not erased by these changes.

Before selecting a new execution path for scientific training, freeze a new
source-bound bounded diagnostic and compare the exact relevant reference on
the target runtime: all outputs/gradients, model and optimizer updates,
schedule/cursor/RNG and serialized repeat/resume. Measure warmed capture-off
throughput separately from cold preparation, full-development time, peak memory
and allocated-device cost. Do not loosen an existing tolerance to pass a probe.
Only accepted native evidence and an explicit new recipe/release may enable
formal training. This code change launches no job and creates no training RML
or Replay-Ready claim. The [implementation verification](../../experiments/modern2d_model_integration/execution_optimization_2026-10-22.md)
records executed checks and remaining evidence gaps.
