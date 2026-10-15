# Desktop logging and execution integration — 2026-10-15

## Reviewed sources and disposition

Server base: `215f697f001cc4d14fb217dea6fcee7cea64d330`.
Reviewed desktop log commit `853ac0ed` and execution refinement
`48c3bdded703931b0fce8a610f09d9a40e8bc1bf`. Both are reasonable shared
infrastructure updates and were selectively adapted, not branch-merged.
Server implementation/test commit: `28114be9`.
The [logging contract](TRAINING_LOGGING.md) owns fields/cadence; the
[execution refinement](GPTRANS_EXECUTION_REFINEMENT.md) owns mechanisms and
upstream qualification limits. This record owns server integration differences.

This is not a new GPTrans-T architecture, pretrained model, scientific result,
MAE gain or measured server speedup. The same backbone/trainable tensors,
objective, FP32/noTF32, BS128, sampler, schedule and EMA policy are retained.

## Integration decisions

| Area | Server integration |
|---|---|
| Standard telemetry | Add shared `training_logging.py` and contract-level `logging` metadata. Epoch-only is the builder default; explicit seconds/batches cadence is validated before data/GPU admission. Canonical evidence remains separate. |
| Reporting order | The existing loop calls progress reporting after required persistence succeeds. Existing metrics, step/exposure, selected state, LR, invocation wall and ETA are printed, with no extra evaluation, tensor synchronization or RNG draw for telemetry. |
| Kaggle relay | Adapt the existing tuple-worker runtime rather than replacing its allocation/retention/barrier logic. Forward complete arm-bound structured lines from each child, retaining partial-line offsets and final draining. No remote operation was performed. |
| Selected EMA view | Select synchronously inside the existing evaluation view, then restore Live even on failure. Retain server's separately recorded Live evaluation for bounded modes. |
| Checkpoint validation | Registered sessions use the synchronous snapshot builder; the existing publication owner validates the complete model/optimizer/EMA/scheduler/cursor/RNG tree once. Public snapshots and generic callbacks still validate independently; no success boolean bypass. |
| Local-bond accumulation | Apply in-place accumulation to `gptrans_capacity.py:BondLocalBlock`, the actual server owner. Do not add a duplicate desktop `gptrans_bond_local.py` factory or replace campaign construction. |
| Native safety-fill reference | New GPTrans releases require the extra training-only two-step fill-on artifact and source/recipe/runtime/observations envelope, matching the accelerated complete-state fingerprint. Missing/copied/tampered proof fails closed. |
| Profiling | Retain server's cached execution/provider, passive spans, native proof and per-layer observation. Add builder/publication/finite-validation ranges without rebranding their timings as a throughput measurement. |

The server's historical bounded-pair traces already store training MAE in eV.
The logger therefore receives the declared observation unit when restoring those
traces; normalized shared traces convert once using frozen training-label std.
Without that adaptation, the upstream default would multiply eV values twice.
Fresh callbacks retain the normalized metric ABI. Tests cover both stored units.

Campaign reference acceptance now reuses `_recipe_configuration`, rather than
reimplementing metadata stripping and rejecting valid optional logging/profiling.
Frozen source pins, contract-byte hashes, target/role identity and comparison
checks remain unchanged. No historical record was rewritten to add logging.

## Preserved boundaries

- Existing four-arm campaign jobs/prefixes use frozen source
  `7e1aac51a852a0b7602cc9f0f45aa83be9d12c5a`; no source migration or successor
  was submitted. Their sixty-pass horizon, ten-pass chunk publication and
  actual-state resume/resource gates are unchanged.
- Server K1 cached providers, Clean-state actions, profiling and job custody
  were not replaced by desktop workflow state. Desktop jobs remain independent.
- Keep existing atomic transaction journal/rollback, complete allocation cost,
  selected/resume schema checks and role/evidence/replay gates. Telemetry does not
  grant scientific acceptance, transfer eligibility or Replay-Ready.
- New source packages require fresh matching release/native qualification. The
  extra safety-fill reference is scratch qualification cost, not formal exposure.
  Renaming an ordinary repeat cannot satisfy the producer-bound proof.
- Desktop synthetic/local-CUDA results remain upstream provenance. No default
  T4/A100/DCU speed or quality guarantee is inferred. The closed desktop synthetic
  diagnostic/its RML corpus were not copied as server evidence or a new runner.
- Existing unrelated user/parallel edits and RML-derived files were excluded.

## Server verification

Project `.venv/Scripts/python.exe`, Torch2.7.1+cpu; CUDA/HIP/ROCR visibility `-1`.
Only elementary synthetic graphs, fresh random weights and mock native metadata;
no real molecular caches, trained checkpoints, protected roles, remote jobs,
formal training, molecular inference or native accelerator measurement.

Independent Luna invocation: **229 passed, 0 skipped**, 46.22s, covering existing
core/session/runner/recipe/loaders/recovery/commit/cycle-release/profiler suites.
`tests/test_modular_experiment_runtime.py` does not exist; it was explicitly
omitted, not called a passing test or substituted native qualification.

Parent-scoped verification:

- Upstream logging/refinement plus server recovery: **81 passed**, 23.17s.
- Server unit-aware telemetry, finite validation, runtime spawn failure,
  portable-resume/profiler proof and static dependency closure: **52 passed,
  2 skipped**, 117.18s. Both skips are explicitly CUDA-only finite-check tests.
- Server bounded cases retain Live/EMA observations, eV units, durable checkpoint
  identity and exact continuation; persistence failure emits no successful epoch.
  Complete checkpoint validation still rejects nonfinite optimizer/scheduler/
  cursor state, and native fill-on proof tampering remains rejected.

Package/release/family-output/modular/kernel regression: **216 passed, 3 skipped**,
108.99s. Skip sites retain host-symlink and explicitly authorized real-input
guards; no real-input qualification was supplied. A requested standalone
`test_experiment_execution.py` file was absent: the first invocation collected
no tests, then the five actual owning suites were executed instead.

Documentation/reuse-routing checks passed **13/13**. A separate actual pair-runtime
mock completed both phases and confirmed final structured lines were relayed
once per arm: **1/1**. The non-overlapping accepted groups therefore cover
**592 passed, 5 skipped, 0 failed** on this server. Scoped `compileall` and
`git diff --check` passed. No full-corpus RML rebuild or cross-machine migration
audit was needed for this code-only integration; user-derived files were not
adopted. No remote submission, continuation, monitoring change or GPU test ran.
