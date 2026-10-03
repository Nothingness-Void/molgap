# Reuse entrypoints

Paths below are relative to the selected checkout. This is a routing index,
not a second copy of contracts, dataset identities, thresholds or live jobs.
Verify APIs against the checkout; missing capabilities remain unsupported.

For a new family or addon, start with the [experiment quickstart](../../../../docs/operations/EXPERIMENT_QUICKSTART.md),
then use this file to select the smallest existing callable. Read only the
relevant detailed contract after the owner is selected.

## Common experiment plumbing

Start with the [addon guide](../../../../docs/operations/EXPERIMENT_ADDON_GUIDE.md)
and read the [CLI reference](../../../../docs/operations/EXPERIMENT_CLI.md) only
for the operation needed. A direct call to the same library is equivalent.

| Need | Existing owner / callable | Boundary |
|---|---|---|
| Identity | [experiment_spec.py](../../../../src/molgap/experiment_spec.py): `ExperimentSpec` | Static registry; do not invent compatible family/mode names. |
| Prospective plan | [experiment_prospective.py](../../../../src/molgap/experiment_prospective.py): `plan_prospective`; [RML plan](../../../../src/molgap/research_memory/plan.py): `plan` | Spec-bound path when supported; existing direct RML API for scoped adapters. Freeze actual authority and source, not placeholder hashes. |
| Source archive | [experiment_package.py](../../../../src/molgap/experiment_package.py): `build_experiment_source_package` | Explicit source allowlist; retained checkpoint/prediction input bundles are not automatically source packages. |
| Variant preparation | [experiment_workflow.py](../../../../src/molgap/experiment_workflow.py): `prepare_experiment_release`; [experiment_staging.py](../../../../src/molgap/experiment_staging.py): `stage_release_inputs`, `UploadArtifact` | Shared local plan/package/layout/release path; read [configuration](../../../../docs/operations/EXPERIMENT_WORKFLOW.md). No platform POST or compute authority. |
| Release inputs | [experiment_preflight.py](../../../../src/molgap/experiment_preflight.py): `check_release_inputs` | Selected imports, packaged recipe hashes, finite frozen CPU initialization and optional upload bindings; no model execution or scientific release. |
| Local receipt | [experiment_launch.py](../../../../src/molgap/experiment_launch.py): `build_launch_receipt`, `reconcile_platform_response` | No network submission or automatic platform authentication. |
| New-protocol family outputs | [experiment_family_workflow.py](../../../../src/molgap/experiment_family_workflow.py): `RunContext`, `FamilyOutputSession`, `inspect_output`, `check_acceptance_plan`, `build_verified_terminal_descriptor`, `close_verified_outputs` | Read [output integration](../../../../docs/operations/EXPERIMENT_FAMILY_WORKFLOW.md); one-time owning-trainer hooks, frozen metric/endpoint requirements and receipt binding. Mechanical verification is not scientific acceptance or replay readiness. |
| IO / digest | [RML trace](../../../../src/molgap/research_memory/trace.py): `atomic_write`, `file_digest`, `json_bytes` | Integrity primitives, not scientific acceptance. |
| Terminal translation | [experiment_terminal.py](../../../../src/molgap/experiment_terminal.py): `translate_terminal_descriptor`, `execute_terminal_descriptor` | Translation alone does not finalize evidence. |
| RML closure | [terminal_wiring.py](../../../../src/molgap/research_memory/terminal_wiring.py): `close_terminal_multi_arm` | Existing per-arm transactions; no fabricated missing evidence. |
| Monitor state | [server_control.py](../../../../src/molgap/server_control.py): `LocalServerControlStore` | Server-owned exact job identity, durable idempotent event; not candidate selection. |

## Training / construction

For repeated GPTrans or EdgeState variants, read only the selected family
section of [the standard workflow](../../../../docs/operations/EXPERIMENT_WORKFLOW.md).
Use `gptrans_screen_adapter.gptrans_screen_arguments` for its explicitly
supported V5 input arms, or `edge_state_screen_adapter.bind_edge_state_screen`
for base/depth/K1 primitive wiring. Neither creates a new scientific recipe,
qualifies an arbitrary registered addon or redirects historical trainers.

Read the relevant [EdgeState adapter](../../../../docs/operations/EDGE_STATE_ADAPTER.md)
or [K1 compatibility](../../../../docs/operations/K1_ADAPTER.md), then the owning
experiment's actual trainer and tests. Reuse
[edge_state_training_core.py](../../../../src/molgap/edge_state_training_core.py)
where its recipe applies. Do not migrate a frozen scientific implementation to
a newer factory merely because they share a family name. The
[integration review](../../../../docs/operations/shared_experiment_verification.md)
separates construction/loader coverage from training qualification.

## Frozen checkpoints and saved artifacts

The shared [experiment_runner.py](../../../../src/molgap/experiment_runner.py)
is **not** a frozen-inference runner. These retained implementations are useful
patterns with experiment-specific assumptions, not generic launch commands:

- [k1_portability_audit.py](../../../../src/molgap/k1_portability_audit.py):
  hash-bound checkpoint/model and accepted-role graph loading.
- [k1_relation_intervention.py](../../../../src/molgap/k1_relation_intervention.py):
  temporary intervention hooks, inference chunks and unchanged-weight checks.
- [k1_relation_audit.py](../../../../src/molgap/k1_relation_audit.py):
  independent acceptance of retained prediction chunks.
- [k1_relation_intervention_records.py](../../../../src/molgap/k1_relation_intervention_records.py):
  saved-tensor comparisons and observed/logical identity reconciliation.
- [k1_relation_audit_records.py](../../../../src/molgap/k1_relation_audit_records.py):
  `prepare_no_train_terminal` translates the accepted two-role relation audits;
  reuse only when its fixed role/cost scope matches, then use RML finalization.

Inspect modes, checkpoint layout, transform, role ranges and helper signatures.
Reuse compatible functions; when values are hardcoded, parameterize the small
shared behavior without changing the original caller. Never execute these old
entries against their bound paths just to start a different experiment.

## Platform operations

Actual submission stays outside the common CLI. Read the selected platform's
workload skill and [platform routing](../../../../platforms/README.md).
For Kaggle, use the [owning adapter](../../../../platforms/kaggle/README.md).
For new ExperimentSpec source packages, run `check-release` with the actual
entry script and staged inputs; pass the report as `--release-report` to the
platform adapter and retain `--response-output`. The adapter repeats selected
checks before POST. Bind the returned ref/URL and version; `submission_unknown`
requires authoritative reconciliation before retrying.
Do not read all platform instructions or credentials when only one is needed.
For the new family output protocol, the Kaggle retriever in
`platforms/kaggle/retrieve_family_outputs.py` streams only manifest-bound files.
Reconcile the actual attempt and pin its small manifest first; historical
formats still use their owning retriever, not this profile as a fallback.

Use explicit authorized credentials without printing them. Check installed SDK
signatures before making a new wrapper. In the tested Kaggle SDK, status accepts
a version-suffixed identity but code pull uses the slug; do not assume every
endpoint accepts the same syntax. Verify the version separately, and reconcile
the returned identity with the frozen release rather than resubmitting.

## Cheap verification

From the repository root, using its configured Python:

```powershell
.venv\Scripts\python.exe -m molgap.experiment_cli --help
.venv\Scripts\python.exe -m pytest -q --noconftest tests/test_experiment_reuse_skill.py
```

Choose the owning component's focused tests for actual code changes. Do not
rerun every historical suite or a model smoke test to validate documentation.
For an evidence change, also run the existing `molgap.research_memory validate`
and `check --frozen`; rebuild only when the source evidence changed. Help/schema
success, intact links and synthetic tests are not release authorization.
The terminal-wiring entry accepts explicit `trajectory.json` and `terminal.json`
paths; do not pass a plan directory to `terminal-pipeline`.
