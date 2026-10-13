# Shared Recovery and Speed Audit

Engineering scope: K1/GPTrans common wheel. No molecular input, protected role,
remote operation or scientific conclusion is needed for these synthetic tests.
The execution contract remains V5; source hashes bind new packages. Historical
and active packages/checkpoints must not be migrated by editing their hashes.

## Repairs

| Defect | Shared owner / correction | Regression |
|---|---|---|
| Deleted AdamW moment accepted | `family_training_session.py`: acknowledged state-ID list, full unconditional coverage and required live base moments | Deletion rejected before restore; legitimate conditional SSMA lazy state accepted |
| Unloadable selected checkpoint accepted | `family_training_outputs.py`: compare selected/resume key, shape and dtype schemas | Rehashed malformed key/shape/dtype blocked for both families |
| Selected/trace published ahead of failed checkpoint | Same output owner: stage, hash-bound journal, rename backups, checkpoint-last commit and validated recovery | Failure before journal, at selected/trace/checkpoint and after commit; exact continuation and whole-bundle recovery |
| Crashed interval omitted from measured cumulative cost | `family_training_runner.py`: start/end invocation marker and acknowledged-checkpoint digest | Interrupted/stale/unbound intervals remain unknown; other-context ledgers rejected |

Recovery edits only producer output files named in a valid context/hash-bound
journal. It never truncates historical canonical evidence or repairs corrupted
hashes. Backups live temporarily in `session_pending/`; they are renamed, not
copied. Mid-pass checkpoint frequency/format translation still has one owner.
Within one exclusive recovery invocation, each visible artifact is hashed once;
commit-point and rollback decisions reuse that verified result, not a hash cached
across invocations. Journal bindings and the full-bundle checks still execute.
Files in an active transaction must not be consumed by another concurrent reader.
The platform must retain these paths when transferring an interrupted output
directory; cherry-picking only `last_checkpoint.pt` loses bundle recovery.

## Safe Operation Removal

1. Metrics: the old session restacked and recopied every preceding batch scalar
   at each checkpoint. The private CPU prefix now receives only the new tail;
   snapshot tensors do not alias that prefix. Five scalars split2+3 transfer as
   2+3, not2+5. Dtype and sequential Python addition remain exact; no tensor sum.
   For the first3906-step pass with saves every128 steps, periodic copies alone previously
   copied59520 scalar values, versus at most3906 with the prefix. This is a byte/
   operation count, not a measured total training gain.
2. GPTrans distance propagation: the frozen model's20-iteration shortest-path
   routine expands a frontier after the last distance write, then discards it.
   The prepared backend performs19, not20 `bmm` calls. Cap1 performszero, with
   identical integer codes. The original `gptrans.py` source/hash is untouched;
   new execution packages use the reviewed prepared adapter, not a slow switch.
3. K1 layout transport: the two private index/mask copies previously requested
   blocking H2D. They now request non-blocking copies on the owning forward's
   current stream. The source tensors are immutable and not shared with a cache;
   CPU layout values and a separate native synthetic optimizer-inclusive equality
   test pass. This is not async checkpoint IO or a model arithmetic change.

Model/loss, AdamW arithmetic, gradient clipping, EMA update order, dropout/BN,
FP32/noTF32, sample membership/order, physical batch, schedule, selection and
checkpoint cadence remain unchanged. Callback publication semantics and the
optimizer participation field are explicit recovery-interface changes.

## Remaining Candidates

The [accepted frozen A100 profile](../../experiments/pcqm_k1_colab_execution_profile/attribution.md)
found forward/backward dominant under its two-forward sample, not data loading.
The [local input-pipeline diagnostic](../../experiments/pcqm_input_pipeline/attribution.md)
did not establish stable bucket gains. These are distinct retained execution
conditions, not a profile of this new single-forward session. Fixed indexing,
persistent loaders, bounded development caching, selective transfer and bound
EMA already exist in the shared wheel; no duplicate pipeline was introduced.

| Candidate | Evidence / risk | Disposition |
|---|---|---|
| Cache per-molecule GPTrans shortest-path codes | Pure topology is recomputed each batch and cached development forward; a cache needs graph/source identity, cap, ordering, memory accounting and integer equality | Not added without a bounded frozen test; no duplicate graph cache |
| Project K1 node endpoints before gathering edges | Reduces repeated projection of equal endpoints but changes GEMM shapes/backward accumulation | Not adopted as bitwise-safe; exact output/gradient/optimizer/RNG qualification required |
| Replace K1 padding assembly or fuse projections | Existing layout already reuses indices; altered allocation/GEMM may change reductions | Profile native operator contributions first |
| Remove EMA schema/storage guards | Guards still cost Python work but detect stale bindings | Keep until a replacement provides equivalent fail-closed detection |
| Change checkpoint cadence or async writer | More apparent throughput but changes durability/exposure contracts; async needs immutable snapshots | Not authorized by this repair |
| Change sampler/workers/optimizer/precision | Previous local bucket probe did not establish a stable gain; these change contract or resource identity | No default change |

## Validation and Limits

Parent writes code; Luna runs CPU tests with explicit CUDA/HIP/ROCR`-1` masks.
Tests use elementary synthetic graphs/weights and fault injection, never real
molecule caches, trained assets, development/test roles or remote jobs.
Exact-resume tests cover both families, schedules, tails, EMA and partial metrics.
The shared runtime remains subject to normal optimizer-inclusive native preflight.

No end-to-end speed percentage, MAE or accelerator utilization is measured in
this engineering audit. Operation-count reductions cannot be called an epoch
gain; hash IO and journal fsync/renames add boundary cost. The earlier local
checkpoint timing used Torch2.7.1, not the default2.11 runtime, and did not include
this journal. Preserve that evidence instead of extrapolating it.

## Executed Acceptance

Luna independently executed these checks in the isolated implementation tree
on 2026-10-13/14. Parent wrote the repairs/tests. CPU runs used the primary Windows
checkout's `.venv/Scripts/python.exe`, `PYTHONPATH=<worktree>/src` and explicit
CUDA/HIP/ROCR visibility `-1`; empty masks were not relied upon. Suites overlap;
their counts are not summed as unique tests.

| Check | Result |
|---|---|
| Workflow/packaging regression | 357 passed, 3 skipped, 0 failed |
| Post-change recovery/K1/kernel regression | 107 passed, 1 skipped, 0 failed |
| Final recovery guards, including retained-generation rollback | 25 passed, 0 failed |
| Final-tree output safety/recovery/runner/workflow after hash reuse | 150 passed, 0 failed |
| Separate native K1 transport equality | 1 passed, 0 failed; 14.44s |
| `compileall` and `git diff --check` | Passed |

Exact test invocations, after setting `$python` to that environment:

```powershell
& $python -m pytest -q tests/test_experiment_family_workflow.py tests/test_experiment_execution.py tests/test_experiment_package.py tests/test_experiment_preflight.py tests/test_experiment_workflow.py tests/test_family_full_lifecycle.py tests/test_family_training_recipe.py tests/test_family_training_core.py
& $python -m pytest -q tests/test_family_recovery_guards.py tests/test_family_speed_equivalence.py tests/test_k1_accelerated_execution.py tests/test_k1_execution_reproducibility.py tests/test_family_training_kernel.py
& $python -m pytest -q tests/test_family_recovery_guards.py
& $python -m pytest -q tests/test_family_commit_safety.py tests/test_family_recovery_guards.py tests/test_family_training_runner.py tests/test_experiment_family_workflow.py
& $python -m pytest -q tests/test_family_speed_equivalence.py::test_native_layout_transport_keeps_output_gradient_optimizer_and_rng
& $python -m compileall -q src/molgap tests
git diff --check
```

The native invocation ran separately with device0, a120s process limit and
PyTorch2.7.1+cu128 / CUDA12.8 / RTX5060. It used random weights, two tiny synthetic
graphs and one optimizer update, not real molecular caches or physicalBS128.
It establishes only transport equality in that condition; the default Torch2.11
runtime, full training, MAE, T4/A100/DCU and end-to-end throughput remain untested.
The107-suite skip was the separately executed GPU-only test. The workflow's
`-q` log records three skips without individual reasons; its skip sites guard
host symlink creation and explicitly authorized real-input release. No real-input
release was granted or substituted. Nothing was fabricated to eliminate skips.
