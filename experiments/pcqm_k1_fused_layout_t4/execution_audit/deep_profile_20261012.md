# K1 deep execution audit, 2026-10-12

Later user-authorized native submission preparation is owned by
[the fresh variant contract](../initialized_layout_100k/protocol.md). Its explicit
Torch2.11.0 pin is a native qualification candidate, not a retroactive change to
the Torch2.7.1 source snapshots or scratch results below. Original default pin
remains2.7.1; both reject a mismatching observed runtime. No2.11 GPU equality is
claimed by local tests.

## Scope and finding

Follow-up to [initial scratch checks](synthetic_checks.md), authorized for
execution optimization. No molecular cache, retained checkpoint, development
or protected role, scientific training, RML publication or remote operation.
Reuse the owner model/step/optimizer, CPU layout and profiler; do not repeat
rejected pre-gather/pre-padding projections or replace the trainer.

PyTorch2.7.1 local determinism also enabled uninitialized-memory safety filling.
This initializes temporary storage before kernels overwrite it. The
[PyTorch documentation](https://docs.pytorch.org/docs/2.7/deterministic.html)
allows disabling that fill only when operations never read uninitialized input;
deterministic algorithms must remain enabled. This is a safety-policy change,
not permission to skip initialization, zeros, finite checks or deterministic
scatter, or to enable TF32/AMP. It is not qualified for arbitrary new/custom ops.

Three-step CPU operator records observed:

| Path | `aten::fill_` calls | Self CPU milliseconds |
|---|---:|---:|
| Original | 6663 | 60.2250 |
| CPU layout | 6450 | 49.4463 |
| Initialized layout | 408 | 3.6077 |

Remaining legitimate fills are preserved. CUPTI initialization failed with
`CUPTI_ERROR_INVALID_DEVICE`; device activity was unavailable. These are CPU
operator records, not GPU kernel times or GPU-busy utilization. Python profiling
also exposed scatter/autograd dispatch; the data-free probe cannot attribute
real loader wait, host/device overlap or full allocation cost.

## Controlled scratch results

RTX5060, Torch2.7.1+cu128, FP32/noTF32, BS128, 3068 nodes/6392 directed edges,
native AdamW defaults, original single-forward L1 and clipping. Nonexclusive
desktop; five warmup/24 measured steps per case per round, three order rotations.
Timer synchronizes block boundaries, not every step. Final measurement calls
the registered owner's `initialized_layout` step, not a duplicate training loop.

| Path | Median seconds/step | Time reduction vs original |
|---|---:|---:|
| Original | 0.096553 | Reference |
| CPU layout | 0.089746 | 7.05% |
| Initialized layout | 0.074319 | 23.03% |

The new path reduces time17.19% versus CPU layout alone; use the raw times rather
than extrapolate whole-epoch speed. The earlier isolated memory-fill probe gave
about25.77% reduction versus original; it did not yet call the registered step.
Neither short measurement establishes T4/A100 gains, dataset throughput or MAE.

One-step predictions/loss, gradients, model state and CPU/CUDA RNG matched
exactly. Eight successive updates across full, ragged, small and edgeless graph
batches matched reference metrics, gradients, BN/model state, AdamW moments and
RNG fingerprints exactly. Mixer return weights were nonzero to exercise K/V.
Scratch model/AdamW/cosine/RNG serialization and the next step matched exactly.
Full40epoch and complete-epoch resume lifecycle tests used small CPU fixtures;
they are infrastructure tests, not scientific training or native qualification.
Focused regression:314 passed,1 deselected. The deselected A100 historical
payload-byte test still lacks its ignored `selected.pt` in this owner checkout;
that unrelated historical artifact verification was not completed here.

Additional source-package, family-output, launch, preflight, runner, Spec and
execution-registry regression: 514 passed, 5 skipped. Three symlink checks were
unavailable on this Windows host; two real-shard checks had no explicitly
authorized input binding. These suites overlap with the focused regression;
do not add their pass counts together or claim real-shard qualification.

## Shared implementation and use

- `training_reproducibility.deterministic_memory_fill_context`: process-global
  temporary setting with exact-bool/fail-closed checks and exception restoration;
  only for exclusive execution workers with separately qualified operations.
- `k1_screen_training`: native-optimizer `initialized_layout` step, frozen policy,
  exact multi-step reference fingerprints and a stronger native admission gate.
  Original epoch loader, schedule, exposure, development/selection and checkpoint
  pipeline are unchanged; all historical mode policies remain intact.
- Spec/execution registry: `k1_initialized_layout/1`, K1 family/version2 only,
  no extra addon or caller-controlled execution switches. Additional runtime
  checks are not RML/evidence-schema changes or training authority.

Prepare a fresh declaration through the existing recipe builder, rather than
edit an old release or reuse old prospective/job identities:

```python
from molgap.experiment_execution import build_family_recipe

recipe = build_family_recipe(
    ("neural_atom_k1", "2"), addon="k1_initialized_layout",
    source_idx_sha256="<real-frozen-row-digest>",
    target_sha256="<real-frozen-target-digest>",
)
```

The executable policy pins native AdamW/foreach=None, CPU layout, original
epoch loader and Torch2.7.1. Safety fill is disabled only within TRAIN steps;
clean evaluation and model construction retain their prior behavior. Unsupported
Torch versions stop; no automatic library installation or upgrade is performed.
Native preflight additionally compares two actual TRAIN fixture steps against
reference, including moments/gradients/RNG. Missing or summary-only comparison
records, mismatched reference observations or non-exact resume reject admission.
Ordinary release/source/input/role/cost/native-runtime gates still apply.

## Evidence and disposition

[Final scratch result](initialized_layout_result.json) and
[operator records](initialized_layout_operators.json) retain the observations
and tested source hashes. Parent Git source is `b8f95004` with explicit dirty
source hashes, not an immutable released package. Exact tested source snapshots
and full profiler/Python traces remain in ignored local custody:
`platforms/_records/local/staging/k1-initialized-shared-profile-20261012/`.
The final probe used60.55 local GPU-wall seconds, excluding interpreter startup;
GPU-busy time and accelerator quota are unknown. No trained checkpoint is
published. This is not a portable or replay-ready scientific trajectory.

Opt-in reusable implementation; no default or production adoption. No new
whole-dataset run was launched. A native platform qualification and complete
pipeline timing remain required before any full-run saving or quality claim.
