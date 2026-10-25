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

The first CPU recovery, kernel140726588/version1, ended ERROR. This task's
adapter imported the package verifier, which eagerly imported NumPy, before
running the platform dependency installation. The later PyG/Torch import hit
a NumPy random-extension ABI mismatch. The frozen training source is unchanged;
the adapter had deviated from the working platform bootstrap order. Version1's
before/after NumPy versions were not captured, and its raw role observation
retains unknown partial membership rather than inventing a zero-read receipt.

The separately frozen version2 installs the same dependencies before scientific
imports, following the existing platform owner. Actual execution confirms those
modules were absent before installation, loaded/installed NumPy1.26.4 agrees,
and the complete original TRAIN loader succeeds. It recovers500,000 finite,
ordered float32 labels with SHA256
`cef6fbb56a546b1808b2a42e94b8a08354d23928512be23de9f1c8b8f66f3676`.
Only2,000,000 label bytes and4,790 data-manifest bytes were downloaded as payload;
small receipts and source reconciliation are separate. The4MB deterministic
index file was reconstructed locally and matched the remote digest, not downloaded.

Both [failed-attempt receipts](target_recovery/reconciliation_attempt_001/failure_reconciliation.md)
and [successful validation](target_recovery/reconciliation_attempt_002/final_validation.json)
are retained. Version2 wall306.038s and process-tree CPU814.829s are separate
measurements; its worker alone used25.952CPU seconds, with dependency/bootstrap
children accounting for the remainder. Version1 wall257.769s and process-tree
CPU784.440s remain charged as failed work. Neither attempt used an accelerator,
model inference or training. Exact source, data and version bindings come from
the receipts; strict training qualification remains a subsequent operation.

The evidence identifies missing producer/consumer compatibility and an incomplete
historical proof contract. It does not show that a large new framework or another
hundred synthetic tests is needed. Qualify the exact frozen recipe through its
owning acceptance path, and reserve strict replay claims for real retained proof.

## Real replay projection incident

After exact TRAIN proof recovery, the actual acceptance invocation completed both
native proofs as VERIFIED and the candidate comparison as STRICT_CAUSAL/VERIFIED.
It still returned BLOCKED after77.044seconds because RML excluded both traces as
`no_compatible_candidate_reference_trace_pair`. The strict identities differed
only in `optimizer_identity`, exactly the frozen weight-decay intervention.

The shared replay selector treated optimizer identity as an unconditional matching
field, even though the hash-bound comparison explicitly qualified that difference.
Commit`af154ac02b7c6d9202c37509d29e16b20969c7af` repairs only this projection. It
binds an optimizer grouping scope to the exact accepted comparison, both actual
optimizer values and the observed physical pair. Other strict fields remain in
the key. Unknown/mixed interventions and frozen metadata that self-declare an
exemption do not obtain this scope. No frozen trace or scientific result changed.

The existing selector on the real canonical records changed from two excluded
singletons to one compatible pair. This read-only reproduction is separate from
the subsequent full acceptance result, recorded by the original tail owner in
`analysis/target_recovery_20261025/qualification_20261025.md`.

The actual rerun exited0 in11.898seconds: strict validation and both independent
replay records VERIFIED, capability complete, no exclusions. Candidate comparison
is STRICT_CAUSAL/VERIFIED; the accepted reference is NOT_APPLICABLE. Reentry used
CANONICAL_READ_ONLY, with no training, inference or network access and31 existing
files unchanged by acceptance. Rebuild and frozen-index check then passed. The
original negative decisions remain in force; READY and a new scientific decision
were not evaluated by this verification.

The new projection supports one exact accepted optimizer pair. Multiple candidates
sharing a reference fail closed; it does not silently invent reference copies or
weaken other matching fields. Internal selector helpers consume validated records;
they are not independent evidence authenticators.
