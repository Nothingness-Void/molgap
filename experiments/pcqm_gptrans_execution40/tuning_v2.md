# Optional execution tuning layer

Implementation record, 2026-10-12. This is infrastructure maintenance, not a
new model result, runtime certificate, training trajectory or Replay pair.

The [subsequent deep audit](deep_execution_audit.md) owns the v3 extension;
this file preserves the earlier v2 implementation record and its test count.

## Evidence and scope

The [accepted delivery result](decision_terminal.md) motivated two execution
questions: dual-worker CPU contention and repeated development input preparation.
The measured evaluation share supports prioritizing the latter; it does not
prove that CPU preparation caused all evaluation time. The v1 backend and native
trainer remain byte-unchanged, preserving their archived source identities.

`src/molgap/gptrans_execution_tuning.py` layers over the existing v1 context.
It reuses the native trainer, epoch sampler, prepared layout, evaluation,
optimizer/EMA implementation and checkpoint hooks. No epoch loop is copied.
Its scope is the same local-bond GPTrans model and fixed contiguous development
role as execution40, not arbitrary GPTrans variants or new datasets.

## Added behavior

- Explicit CPU thread/loader-worker budgets for independently isolated models.
  The initial configuration declares one compute thread and one training worker
  per model, zero development workers, two models and four aggregate CPU units.
  This is a candidate configuration, not a measured optimum. The launcher must
  verify available CPU quota and the actual number of independent processes;
  this process-local context cannot enforce another process's settings.
- Bounded CPU retention of collated development batches and prepared layouts.
  Live/EMA and later evaluations reuse inputs, never predictions or weights.
  The initial limit is 256 MiB of retained tensor storage **per model**, not total
  process RSS. Python objects, loaded datasets and allocator overhead are extra.
  Overflow explicitly falls back to complete streaming evaluation.
- A cache is published only after a complete, row-validated pass. Shallow PyG
  storage-wrapper copies isolate device transfers; cached CPU tensors are
  borrowed read-only. Stochastic dataset transforms are rejected. The owning
  dataset must remain immutable for the context lifetime.
- Distinct backend identity binds every tuning option. Thread settings,
  environment and patched methods are restored even if worker cleanup fails.
  Caches are released at context exit.

FP32/no-TF32, BS128, model, loss, schedule, target transformation, sampler,
development row order/count, live/EMA evaluation frequency and CPU FP32 metric
reduction are not intentionally changed. One main CPU RNG draw per evaluation
is preserved even for warm cache hits. No GPU numerical equivalence is claimed
from CPU tests or from an earlier runtime certificate.

## Existing trainer integration

Use the context inside each isolated worker, around the native preflight and
training calls. Do not nest it with another execution context. Import-time CPU
environment configuration belongs to the launcher; `environment()` supplies
the corresponding OMP/MKL settings before worker launch. The context also sets
PyTorch intra-op threads in the worker. No inter-op pool or CPU affinity policy
is silently changed.

```python
from molgap.gptrans_execution_tuning import ExecutionTuning, tuned_backend

tuning = ExecutionTuning()  # freeze the actual options before remote execution
with tuned_backend(native_owner, tuning, timing_root=timing_directory) as report:
    # Call the existing owning preflight/training functions, not a copied loop.
    ...
# Persist report with existing atomic_json in the owning adapter.
```

The yielded report contains tuning identity, phase timing, cache hit/fallback
counts, retained storage size and `remote_qualified=false`. Serialize it using
existing atomic IO; a Python context alone does not publish remote artifacts.
The execution40 worker is intentionally not changed or resubmitted.

## Local verification

Executed with the project interpreter:

```text
python -m pytest -q --noconftest tests/test_gptrans_execution_tuning.py tests/test_gptrans_execution_backend.py
62 passed; 2 expected pin-memory warnings on a CPU-only runtime
```

Synthetic CPU checks cover cached/streaming live and EMA payload equality,
main RNG equality, frozen native 100K sampler order at epochs 0/1/39, row/count
rejection, interrupted and oversized caches, device-transfer isolation,
transform rejection, worker-cleanup faults and restoration. They load no
retained MolGap models, checkpoints, molecular datasets or protected roles.
Worker spawning and real GPU performance remain unqualified.

## Remote qualification still required

No new cloud job was submitted and no speedup was measured for v2. Before reuse
in a scientific screen, freeze a bounded execution-only comparison against v1,
with accepted fixed inputs, identical initialization/steps and actual allocation
cost. Separate cache-on/off from CPU-resource changes rather than attributing a
combined gain to one mechanism. Include cold and warm evaluation, live/EMA
payload/row checks, synchronized phase times, optimizer/EMA/RNG resume state,
process RSS and GPU memory/utilization. Declare equivalence criteria and a cost
ceiling before compute; do not tune them after observing results.

Use the existing family/platform release and acceptance components, resolve the
applicable prospective reference semantics before release, and hand monitoring
to existing Luna B. Do not repeat the historical missing-reference binding or
claim training Replay readiness for a profiling result. No automatic forty-pass
rerun, new seed or scientific promotion is released by this implementation.
