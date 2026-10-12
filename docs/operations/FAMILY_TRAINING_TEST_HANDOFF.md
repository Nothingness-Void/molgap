# Independent Test Handoff

## Scope

Worktree: `C:/Users/17449/.codex/worktrees/family-training-core/molgap`.
Branch: `codex/infra/family-training-core`, desktop base `e351689b`.
The external run tested the then-uncommitted implementation tree, not its old
HEAD. Verify actual branch/HEAD/status for any later rerun. Integration preserves
unrelated primary-desktop changes.

**CPU_WIRING_ACCEPTED**, based on the user-supplied external rerun report below.
Parent did not execute these tests. Native qualification remains unexecuted.
This acceptance applies to the described uncommitted tree, not its old HEAD or
an immutable release package; the report supplied no tested-tree digest.
Parent writes code only.
Test/inspect only: no production edits, real-repository commit/push,
submission, real molecular inputs/retained weights, protected roles, GPU/DCU,
server changes or RML mutation. Temporary synthetic fixtures and fixture Git
commits are allowed. Native resource/role authorization is a separate request.

## External Acceptance

The external tester reported all five commands in the Commands section executed
with CUDA/HIP/ROCR devices hidden and the worktree's `src` on `PYTHONPATH`:

| Check | Passed | Failed | Skipped | Seconds | Exit |
|---|---|---|---|---|---|
| Suite 1: shared core | 196 | 0 | 0 | 50.83 | 0 |
| Suite 2: family compatibility | 168 | 0 | 0 | 32.92 | 0 |
| Suite 3: workflow/CLI | 875 | 0 | 8 | 514.26 | 0 |
| Compileall | - | 0 | - | 1.1 | 0 |
| Git whitespace check | - | 0 | - | 0.8 | 0 |

The tester confirmed the three defects below resolved. The eight skips were
reported as unavailable historical Colab `selected.pt` fixtures, not substituted
evidence. No real molecular input, retained-weight training, native accelerator
execution, platform submission or RML evidence merge was tested. This result
does not certify native repeat/resume, speed/quality, scientific comparison or
replay readiness. At acceptance the code was still on the implementation branch;
Git integration is separate from test execution. Any later implementation edit
requires applicable revalidation.

## Git Integration Record

On 2026-10-12, implementation commit
`d1b23b0a6ab6d7580f82ec88bdcd16081a188426` was fast-forwarded into local
`molgap-desktop` with user approval. Parent inspected identical implementation
and test trees against the owning branch; no test was rerun. Only documentation
changed after the external rerun. The user's workflow edit was restored without
conflict and remains uncommitted; unrelated untracked content was untouched.
No push, remote submission, model promotion or native qualification occurred.

## Corrective History

User-supplied first-run report on this uncommitted tree at base `e351689b`:
suite 1: 187 passed / 1 failed; suite 2: 168 passed / 0 failed;
suite 3: 865 passed / 6 failed / 8 skipped. Compileall and whitespace checks
passed. These results precede the following corrective edits, not acceptance.

- Resource validation now rejects empty/vendor-only or generic normalized names,
  including a generic observed name under a HIP/DCU declaration.
- Worker module lookup tolerates an omitted key. Only the shared family runner
  may return `PAUSED` successfully, and only in its training phase.
- The CLI's exact static command whitelist includes `prepare-family-training`;
  remote/dynamic-dispatch restrictions remain unchanged.

Negative allocation cases and worker pause/phase/module cases were added.
For future related fixes, the following focused command is a useful starting
point after environment setup; it does not replace all-suite acceptance:

```powershell
& $python -m pytest -q tests/test_family_training_runner.py tests/test_experiment_workflow.py tests/test_experiment_family_workflow.py tests/test_experiment_cli.py
```

## Commands

Run from this implementation worktree:

```powershell
$env:PYTHONPATH=(Join-Path (Get-Location) 'src')
$env:CUDA_VISIBLE_DEVICES=''
$env:HIP_VISIBLE_DEVICES=''
$env:ROCR_VISIBLE_DEVICES=''
$python='D:/文档/molgap/.venv/Scripts/python.exe'
& $python -m pytest -q tests/test_family_training_core.py tests/test_family_training_data.py tests/test_family_training_recipe.py tests/test_family_training_kernel.py tests/test_family_training_loaders.py tests/test_family_training_session.py tests/test_family_training_runner.py tests/test_family_full_lifecycle.py
& $python -m pytest -q tests/test_k1_execution_reproducibility.py tests/test_k1_screen_lifecycle.py tests/test_k1_two_pass_mean.py tests/test_k1_accelerated_execution.py tests/test_gptrans_execution_hotpath.py tests/test_gptrans_execution_tuning.py tests/test_pcqm_gptrans_v4.py
& $python -m pytest -q tests/test_experiment_spec.py tests/test_experiment_execution.py tests/test_experiment_package.py tests/test_experiment_preflight.py tests/test_experiment_workflow.py tests/test_experiment_family_workflow.py tests/test_experiment_cli.py tests/test_experiment_launch.py tests/test_experiment_prospective.py tests/test_experiment_terminal.py
& $python -m compileall -q src/molgap tests
git diff --check
```

Confirm actual test filenames before execution; report missing files and use the
owning test, not a silent skip. Historical full-source pin tests are known to
target immutable old source; report separately and do not update old pins to
obtain green checks. The ignored historical Colab `selected.pt` is absent here;
do not fabricate it. Do not sum overlapping suite counts.

## Required Cases

| Area | Test | Expected |
|---|---|---|
| Scale/family | Both families, 100K/500K/full dimensions | One registered runner/core/session; only explicit policies differ |
| Recipe | Unknown import/field, slow backend, wrong init/mode/schedule/counters | Fail before publication/training |
| Normalization | Preserve frozen K1 values; change a value | Different identity; no runtime recomputation |
| Data | Remount; alter SHA/rows/order/labels; overlap/protected range | Remount identity unchanged; altered input rejected |
| Role omission | Delete dev asset for final-only/preflight | No dev read or invented metric |
| Loader/cache | 0/1 worker, pass/resume, tail, K1 RWSE16, GP layout, cache overflow | Exact order/exposure; private RNG; no mutation/truncation |
| Resume | Four schedules, EMA/no EMA, dropout, partial-pass metrics | Same weights, optimizer, scheduler, RNG and metric order |
| Corrupt resume | Budget/cursor/sample/history/model/EMA/optimizer/RNG change | Fail closed; failed restore rolls back state |
| Update contract | Rehash checkpoint after changing LR, beta/epsilon/decay, foreach/fused, moment shape/count, cosine period/base/minimum/epoch | Reject even with unchanged recipe identity; no live-state mutation |
| Initialization | Correct tensor hash but float64 weights for FP32 model; wrong shape or default dtype | Reject instead of silently converting transported weights |
| Persistence | Selected/checkpoint write failure, including terminal | Exception, cleanup, no reported complete endpoint |
| Output | Synthetic session -> artifact hooks -> inspector | MECHANICALLY_VERIFIED only; other gates NOT_EVALUATED |
| Final-only | Partial last pass, no predictions, final live/EMA hash | Correct exposure; no fake epoch/dev score |
| Semantic tamper | Corrupt context/counters/roles/selection/trace/metrics and rehash manifest | Still BLOCKED, not just a SHA check |
| Native proof | accepted=true only; stale identity; missing/changed repeat/resume-origin files | No training release |
| CPU proof | Empty CUDA RNG or CPU-only provenance | No native qualification |
| Allocation | Declared T4 vs actual A100; RTX5060 vs Ti; generic GPU; CUDA device labelled DCU | Reject; allowed exact/vendor names and explicit A100/DCU resource classes do not select a different algorithm |
| Source | Import outside packaged root or wrong bytes | Reject before formal training |
| Preparation | Committed temporary source/recipe and prospective actions; each platform declaration | Same package/init/prospective staging; no submitter/account invention |
| Prepare failure | Duplicate device/arm, dirty source, reserved path, bad init, partial prospective | No prepared success or successor; retain partial files |
| Worker | Preflight barrier; PAUSED vs complete result | Partial work retained, never promoted |
| Costs/retry | Missing earlier cost ledger; completed retry | Unknown is not zero; retry does not retrain/reload data |
| Boundaries | Registry/default, V5 topology, server, canonical/derived RML | No changes or new scientific conclusion |

Additional preparation/proof/tamper cases require targeted temporary fixtures
or static review beyond the supplied unit file. Do not infer those passes from
a narrow green suite. Native speed/quality and actual platform execution are
unverified until independently authorized and measured.

## Report

Give branch/HEAD/status, exact commands and pass/fail/skip counts, then failures
with file/line and traceback. Separate implementation defects, unavailable
historical artifacts and unexecuted native checks. End with `CPU_WIRING_ACCEPTED`,
`BLOCKED` or `INSUFFICIENT_EVIDENCE`, never full scientific/RML readiness.
