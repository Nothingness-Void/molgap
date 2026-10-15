# Cross-scale V1 CPU acceptance — 2026-10-16 JST

Status: **INFRA_CPU_ACCEPTED**, scoped to the new non-executing comparison layer.
This is not project-wide green regression, real-data comparison admission,
MODEL_FIDELITY_ACCEPTED, native qualification, training release or model ranking.
The two independently reproduced baseline regression failures below remain open.
The [machine record](CROSS_SCALE_INFRA_ACCEPTANCE.json) owns exact pins, cases and
test counts; the [contract/audit](CROSS_SCALE_EVAL_INFRA_V1.md) owns API semantics.

## Delivered files and reuse

Six new reusable modules supply typed draft review, cohort/sidecar guards,
observed curve and native-cost views, checkpoint snapshot/evaluation guards,
historical backtest and future model-fidelity evidence. Four test modules,
one declaration helper and an exact-byte historical JSON fixture supply CPU
proof. `ARCHITECTURE.md` adds their routing pointer.

No existing training loop, optimizer, scheduler, sampler, loader, Session,
Logger, executable Spec wire format, source packager, platform adapter, RML
writer, historical experiment or model implementation was changed. Tests reuse
the existing elementary stochastic session fixture, real shared Session,
optimizer/schedule/EMA/sampler, canonical RML validators and atomic IO owners.

## Test evidence

- Before implementation validation: 264 existing core/data/session/logger/profile
  tests passed (68.26 s), CPU only. This overlaps the expanded run.
- Final new layer: **102 passed, 0 failed, 0 skipped** (14.43 s), after all code
  changes. No native device was visible; Python was the repository `.venv`,
  Torch `2.7.1+cpu`, CUDA unavailable, CUDA/HIP/ROCR masks explicitly `-1`.
- Expanded mixed regression: **1379 passed, 2 failed, 8 skipped** (426.93 s).
  It included the then-95 new tests, so counts must not be added to 102. The
  final seven additional/strengthened cases were validated by the final new
  suite. Existing mechanism contents remained unchanged throughout.
- Both failures were reproduced against an independent `git archive` export of
  original base `63481324aceb807a3f74a08ba2c62bd3e0dcf9ab`, with its source on
  `PYTHONPATH`, the same project interpreter and devices hidden. Neither was
  caused or repaired by this patch.
- Compileall, RML validate and RML frozen checks passed. No RML rebuild,
  trajectory publication, terminal write or scientific outcome change occurred.
- Additional layout/documentation/reuse run: 38 passed and one new-document link
  failure (115.81 s). The two Desktop-only source links were converted to pinned
  commit permalinks; all 13 documentation/reuse tests then passed (2.46 s).
  The 26 repository-layout tests had passed in the first run and were unchanged.
  Entrypoint budgets and staged whitespace checks passed. These counts overlap.

The expanded command selected these existing ownership groups explicitly:
shared-training compatibility; family core/data/recipe/kernel/loaders/session/
runner/full-lifecycle/state/profile/profile-release/recovery/server-recovery/
commit-safety; training logging; experiment Spec/registration/package/preflight/
workflow/resume/family-output/launch/prospective/terminal/CLI/retention/hooks;
comparison-readiness; allocation; Common contract and shared-family registration.
Exact paths are in the machine record. Initially attempted obsolete test names
`test_family_training_outputs.py` and `test_experiment_execution.py` were absent;
those commands collected no tests. Their real owners were tested instead; no
silent skip or artificial passing file was created.

## Required acceptance cases

| # | Case | Result / proof |
|---:|---|---|
| 1 | Nested synthetic cohorts, separate hashes, no overlap/reordered targets | PASS — plan membership/target tests |
| 2 | Remount unchanged; altered bytes fail | PASS — sidecar remount and byte tamper |
| 3 | Equal optimizer steps despite different passes; no epoch alignment | PASS — aligned intersection only |
| 4 | Mid-pass snapshot, read-only state/buffers/RNG, exact resumed state | PASS — shared Session + live/EMA + BN fixture + persistence failure |
| 5 | LR/state/precision/transform/role/sampler differences | PASS — machine-readable confounders; no contextual delta |
| 6 | Slow 100K start, later 500K win; no old control imports | PASS — synthetic reversal and static dependency check |
| 7 | 100K deficit/fidelity gap/incomplete primary is not 500K negative | PASS — independent outcome axes |
| 8 | One seed or sign disagreement is not robust superiority | PASS — seed uncertainty and duplicate rejection |
| 9 | Missing prior invocation/cost/allocation is not zero | PASS — existing RML/allocation owners, idle and conflict checks |
| 10 | Forbidden/tampered role fails before opening tensors | PASS — permit/authority/history spy; hidden protected names |
| 11 | Sidecar version/order/3D/size/padding/batch/edge guards | PASS — byte-only and topology fixture scopes remain distinct |
| 12 | Accepted September early-to-late contraction; cross-role context | PASS — original trace SHA256s and analysis regression |
| 13 | Unknown family/revision/key/source/status cannot become runnable | PASS — draft and fidelity records have no execution/compiler path |
| 14 | Synthetic plan -> checkpoint/evaluation/curve/report; no RML/remote | PASS — owning shared session end-to-end |
| 15 | Existing scoped family/workflow regression | FAIL, pre-existing — 2 failures reproduced on base; no new regression observed; 8 scoped skips retained |

### Open baseline regression findings

1. `tests/test_experiment_registration.py:35`,
   `test_single_family_package_excludes_unselected_trainers_and_addons`:
   current GPTrans dependency closure includes `k1_screen_training.py`, contrary
   to the old test's exclusion assertion. No source allowlist was weakened and
   no test expectation was changed. The owning packaging review must resolve
   the intended dependency-isolation policy before relying on that gate.
2. `tests/test_experiment_cli.py:449`,
   `test_no_remote_or_dynamic_dispatch_in_cli`: the static command-set assertion
   omits already implemented `check-workflow`. Its AST no-remote/dynamic checks
   precede that assertion; no new CLI command was added by this delivery.

The eight skips were six host symlink-permission cases and two explicitly
unauthorized real-shard preflight cases, independently reconfirmed with `-rs`
(8 skipped, 2.13 s). Neither synthetic data nor elevated permissions were used
to turn those native/host checks into passes.

## Remaining gates (not executed)

- **ROLE_BLOCKED:** retained 100K development overlaps 500K training. A future
  common internal-development cohort requires explicit authority outside both
  train sets. Existing 500K internal development may be proposed, not silently
  consumed or relabeled independent. No new split/cache was built.
- Real source-index/target tensor digests not present in retained small metadata
  remain unknown; no graph tensor was loaded to complete them.
- Native checkpoint evaluation, Clean-BN evaluation, real graph sidecar
  preprocessing/acceptance, allocator/runtime certificates and hardware speed
  remain unqualified. V1 read-only evaluation enforces CPU synthetic permits.
- No real model fidelity proof or new family registration exists. Drafts cannot
  compile into executable Specs. Future compilation/admission uses existing
  V5 owners only after separate model/data/resource authority.
- Historical backtests are descriptive and missing native-cost/calibration
  evidence remains missing. No automatic quality stop, STQS rule, seed expansion,
  500K veto, 500K/full release or formal ranking was activated.

## Explicit Desktop transplant

Apply the delivery's single `feat(cross-scale)` commit selectively to the owning
Desktop integration workflow; do not merge the server branch. First align the
reviewed [common mechanism snapshot](SHARED_TRAINING_CORE_SERVER_RETURN.md) and
run `test_shared_training_compatibility.py`. Preserve Desktop's own family
catalog, policies, job custody, source inventories and historical pins. The new
modules call existing common APIs and require no server controller/credentials.

Then run the four new test modules with the Desktop source on `PYTHONPATH` and
CUDA/HIP/ROCR hidden, plus the matching scoped regressions. The historical fixture
is portable: embedded original bytes have their own SHA pins, with no dependency
on a mutable branch name, Temp path or absent Desktop checkout. Host fixed-data
metadata audit may skip if metadata is not installed; do not copy/rebuild caches
to force that optional host check. Desktop application itself was not executed.

The user's `tests/test_research_memory.py` SHA256 remained
`e623abd9b0a410d6c49ccf1d04e2e1ba5bb23237db4dea2a2c8c874f4b9465f1`.
All original dirty research README/derived/local records were preserved and
excluded from the commit. No push, GPU allocation, remote job operation, real
molecular forward, protected-role access or production-model change occurred.
