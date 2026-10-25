# Actual infrastructure incidents — 2026-10-25

This review concerns the completed tail run 140639445/version 1 and the local
diagnostic. It does not infer a training failure from an acceptance failure.
Both tail arms finished their declared 20 passes and retained verified native
proofs. Their scientific negative decision remains unchanged.

## Reproduced acceptance defects

| Incident | Actual failure | Smallest repair and boundary |
|---|---|---|
| Frozen recipe compatibility | Tail source declares `initial_development_check` and `development_reports=['ema']`; desktop recipe/runner rejected them before qualification | Reuse the owning branch's predictor guard and reporting semantics; optional fields preserve old recipes and LIVE selection |
| Comparison intervention scope | Same-run strict qualification hard-coded architecture intervention; this frozen pair changes only optimizer weight decay | Resolve the declared purpose from the archived human input; permit optimizer identity only for declared optimizer comparisons; reject unknown/mixed groups |
| Missing ordered TRAIN target proof | Original target manifest has only development-label identity | Existing desktop repair returns structured BLOCKED and accepts a bound supplemental target-transform asset; new preparation rejects missing TRAIN hash. Actual proof recovery is still separate work |

The first two fixes are commit `9b943850a926627341b8c23f806f3155b42f22aa`.
Recipe/guard/reporting support was reviewed from the exact tail owner `b6114c8a`;
the integration preserves desktop resource cleanup. No frozen recipe, target,
normalization or accepted scientific record was rewritten. Historical mean/std
rounding to the executed FP32 values is preserved, not treated as an invented bug.

Real metadata verification binds source `873c3ac4e8c04f7658d5f1711dfe4e17f7d9eb12`,
archive `ce4326952cc6b379cb577aac9bb090adfe8e290922839e397dd92a38f643a055`,
package `09efe4c0b754b0c7b7f2d1ea8e7a679be3b0a7b2763eb30a8afc9fff131c1e32`
and spec `f5404cb801a792bf57f56dfde0b496e8f21a7ad67b9f22f93a8721cc1d04a994`.
Both actual training-contract hashes match their retained output contexts.
Producer `platform_version=null` remains unchanged; actual version 1 comes from
the physical receipt. The resolver does not fabricate producer metadata.

Peer review found two additional concrete weaknesses in that patch: rebuilding
a recipe dropped its initialization guard, and NaN/shape-invalid retained values
could evade ordinary tolerance comparisons. The compiler now retains compatible
guards and explicitly rejects inheriting them across architecture changes. The
existing runtime guard checks shape, row count, index dtype, finite values and
consistency between MAE and retained prediction bytes before qualification.
The reporting-resume unit test is correctly named an in-memory resume; the
separate real worker test owns serialized checkpoint and recovery coverage.

## Verification actually executed

One focused invocation passed 21 checks in 121.57 seconds:

```powershell
& 'D:/文档/molgap/.venv/Scripts/python.exe' platforms/local/run.py --profile D:/molgap-machine-profile.json --module pytest -- tests/test_family_training_runner.py::test_contract_rejects_counter_changes_and_unknown_fields tests/test_family_training_runner.py::test_actual_session_output_translation_and_cpu_mechanical_inspection[True-neural_atom_k1] tests/test_family_training_session.py::test_live_selection_with_frozen_ema_reporting_survives_serialized_resume tests/test_scientific_pair_input.py tests/test_run_acceptance_integration.py::test_top_level_execute_restores_frozen_generated_metadata -q
```

Working checkout: `D:/molgap-worktrees/k1-scale-attribution-20261025`.
Real CPU kernel/session/output/resume and initial-predictor guards ran. The
strict integration fixture substitutes native admission and comparison identities;
it does not certify a new remote GPU job. A separate read-only real-tail metadata
smoke exercised `resolve_run_acceptance_inputs`, `_frozen_same_run_scopes`,
`resolve_strict_target_transforms`, `read_recipe` and target-statistic binding.
It now reaches the genuine missing TRAIN proof rather than failing on the recipe.

No standalone logs were saved for that invocation/smoke. Their original outputs
are tool session 52096, final chunk f51668, and smoke chunk ce3485. Do not claim
an independent saved log or rerun merely to create one. Subsequent closure logs
are separately retained in `D:/molgap-artifacts/reconciliation/k1-scale-attribution-20261025/`.

The follow-up guard repair used one focused selection of four existing test
instances, not another full suite. The first command had a node-name typo and
exited4 during collection; it is not a passing test execution. Correcting that
literal name produced exit0, `4 passed, 1 warning in 44.68s`, outer wall48.807s.
Exact command/stdout/stderr/timers are retained in
`infrastructure_verification/guard_attempt_001/` and `guard_attempt_002/`.
The warning is the upstream torch_geometric.distributed deprecation. Native
GPU qualification and the missing TRAIN digest are not established by these tests.

## Local diagnostic adapter errors

The first inference adapter preloaded checkout constants before binding the
frozen source path. Its guard stopped before any model forward after 29.8604183
seconds; CPU time was not retained. This was an error in this task's adapter.
The correction reuses the existing frozen BN diagnostic's import discipline.
The separately prospective retry completed all five states in 647.321 worker
wall seconds and reproduced the retained predictions. No broad tests were added.

Two retry-preparation errors also occurred before inference: a reused hypothesis
ID was rejected by the planner, then a Windows default-decoding edit damaged a
Unicode archive path. Unique IDs and an explicit Unicode path/UTF-8 environment
fixed them. These are preparation mistakes, not model negatives or evidence that
the original Kaggle training failed. Their failed attempts did not publish a new
prospective or launch computation.

## Remaining operational scope

The CPU label-export task is a narrow evidence-recovery adapter. Existing T4
submission and paired runtime cannot be presented as a CPU route. It reuses
package verification, frozen shard loading, explicit account credentials and
selective retrieval, with a server session timeout plus an absolute deadline.
It must preserve the original target manifest and normalization. Completion,
receipt, actual role use and strict replay are checked independently; writing
a plausible target hash is never evidence recovery.

The evidence identifies missing producer/consumer compatibility and an incomplete
historical proof contract. It does not show that a large new framework or another
hundred synthetic tests is needed. Qualify the exact frozen recipe through its
owning acceptance path, and reserve strict replay claims for real retained proof.
