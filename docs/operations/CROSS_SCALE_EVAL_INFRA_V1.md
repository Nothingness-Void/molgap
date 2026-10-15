# Cross-scale evaluation infrastructure V1

## P0: reuse versus gap audit (2026-10-16 JST)

Implementation checkout: `molgap-server`, base
`63481324aceb807a3f74a08ba2c62bd3e0dcf9ab`. The human explicitly selected this
checkout and an explicit Desktop transplant, not a branch merge. Inspected
Desktop reference: `ea0c400c34551afe428fda58de97e368f61dfbcf`;
local Desktop ref `f4956019453b75d85347038d46a917fddc9dbdd1` was behind it.
The original worktree was dirty (research README/test edits, derived RML and
platform storage); none of that is part of this delivery.

| Need | Existing owner/API reused | Actual gap / minimum extension |
|---|---|---|
| Updates and exposure | `family_training_core.run_training_loop`, `optimizer_update`, `LoopProgress` | Read-only normalization of observed coordinates; no new loop |
| Schedule and recipe | `FamilyRecipe`, `ScheduleSpec`, `ContractSchedule` | Draft comparison identity, separate from executable Spec; no new scheduler |
| Data and sampler | `DatasetSelection`, `resolve_pinned_manifest`, `load_training_data`, `PassBatchSampler`, `FamilyLoaders` | Cross-cohort membership/role and structural-sidecar validation; no graph rebuild or loader |
| Mid-pass recovery | `FamilyTrainingSession.state_dict/load_state_dict`, `validate_session_resume` | Read-only exact-step snapshot/evaluation guard around retained state |
| Atomic outputs | `SessionOutputs`, `FamilyOutputSession`, `publish_immutable_bytes` | Separate analysis output; do not pretend mid-pass evaluation is a completed pass |
| Logging | `TrainingLogger`, `TRAIN_LOG` | Curve projection reads canonical observations; no new timer/logger |
| Profiling | `FamilyExecutionProfiler`, `ProfileConfig` | Preserve measured phase scopes, unknowns and hardware in analysis |
| Source and executable identity | `ExperimentSpec`, static `TRAINING_ADAPTERS`, existing workflow/package/receipt owners | Draft-only future compilation boundary; no executable schema change or publisher |
| RML | `validate_canonical_trace`, `validate_manifest_trace`, `file_digest`, `repo_local_path`, existing cost/role validators | Hash-bound read-only trace import and independent outcome axes; no RML writer |
| Cost | `research_memory.cost.build_cost_ledger`, `experiment_allocation.AllocationLedger` | Invocation completeness and measured/estimated/missing coverage in read-only reports |
| Historical regression | Desktop accepted scale reassessment and its canonical trace pins | Versioned source-only fixture and eligibility/backtest; no STQS/control integration |
| Future model fidelity | Static family registry / existing CPU sandbox | Evidence record only, no PPGT/ESA/DGT implementation |

`input_pipeline_diagnostic.py`, `scale_transfer_diagnostic.py`, `futility_gate.py`,
`models/REFERENCE_INDEX.md` and the September reassessment are absent from this
server tree. Their applicable Desktop Git objects were inspected read-only.
The missing input diagnostic is not recreated: existing loaders/profiler own
its relevant measurements. STQS and epoch-prefix futility are diagnostic-only;
the new decision path must not import or call either.

### Real fixed-data qualification

Only retained small manifest/acceptance JSON was read, never graph tensors.
Accepted 100K train `[0,100000)` is nested in accepted 500K `[0,500000)`;
its two training topology shard hashes match exactly. However, 100K development
`[100000,150000)` overlaps 500K training; 500K development is
`[500000,550000)`. A strict shared-role cross-scale claim is **ROLE_BLOCKED**.
Source-index/target tensor digests not present in these metadata remain unknown.
No graph shard was opened to fill them. The original graph contract also lists
geometry inventory: only explicitly selected topology is eligible; a manifest
containing geometry is not itself proof of model-input leakage.

A future common internal-development role requires explicit authority and
disjointness from both train memberships. This delivery neither resplits nor
authorizes that role. Historical valid within-contract decisions are unchanged.

### Existing work / evidence boundary

Root [CURRENT_STATE](../../CURRENT_STATE.md) records four server chains paused
at accepted pass10/60; the short diagnostic follow-up was closed. This was a
local record observation, not a remote status query. Desktop jobs, reference
weights and protected roles were not adopted or opened.

Reviewed contracts: [V5](MOLGAP_COMMON_DIRECTION_V5_FINAL.md),
[shared core](FAMILY_TRAINING_CORE.md), [test handoff](FAMILY_TRAINING_TEST_HANDOFF.md),
[logging](TRAINING_LOGGING.md), [outputs](EXPERIMENT_FAMILY_WORKFLOW.md),
[workflow](EXPERIMENT_WORKFLOW.md), [CLI](EXPERIMENT_CLI.md),
[shared inventory/handoff](SHARED_TRAINING_CORE_SERVER_RETURN.md), and Desktop
[scale reassessment](https://github.com/Nothingness-Void/molgap/blob/ea0c400c34551afe428fda58de97e368f61dfbcf/experiments/pcqm_scale_transfer_reassessment/decision.md)
(accepted 2026-09-25),
[RML curve review](https://github.com/Nothingness-Void/molgap/blob/ea0c400c34551afe428fda58de97e368f61dfbcf/docs/operations/DESKTOP_RML_CURVE_REVIEW_20261013.md),
[reference index](https://github.com/Nothingness-Void/molgap/blob/ea0c400c34551afe428fda58de97e368f61dfbcf/models/REFERENCE_INDEX.md)
and [branch roles](https://github.com/Nothingness-Void/molgap/blob/ea0c400c34551afe428fda58de97e368f61dfbcf/BRANCHES.md)
at the pinned Desktop revision. These Git objects were inspected locally; the
links identify the reviewed source, not an extra online or dataset access.

Implementation, CPU verification, native qualification and scientific acceptance
are separate. The [acceptance record](CROSS_SCALE_INFRA_ACCEPTANCE.md) owns test
results, remaining gaps and the explicit transplant instructions.

## P1-P7 operating contract

The implementation is a typed, read-only comparison layer. Drafts never compile,
publish trajectories, prepare packages, submit, resume, monitor or cancel jobs.
The existing Spec v2 and V5 records remain unchanged. Unknown fields/hashes,
unregistered model revisions and unauthorized/protected roles fail closed.

Controlled nested-data comparisons share an explicitly authorized development
cohort outside both train sets and a common step-indexed LR horizon. Diagnostic
cutoffs do not shorten that horizon. Tuned recipes are a distinct system view;
recipe differences never become architecture-only causal effects. New structural
features require separate versioned pure-2D sidecars and CPU acceptance. Node
RWSE16 is not pairwise RRWP. No such cache is generated here.

Curves use observed steps, presentations, unique members, complete passes,
measured wall and allocated-device hours (missing remains null). Epochs alone
never supply unobserved steps. Online train, fixed train probe, live/EMA/Clean
development are separate metrics. A mid-pass snapshot retains its true cursor;
an independent read-only evaluator may not update selection or training state.

100K MAE never vetoes a 500K hypothesis. Invalid implementation, nonfinite state,
corrupt identity, forbidden roles, persistence failure or a declared cost ceiling
can block the evaluated envelope. A cost-truncated primary is INCONCLUSIVE /
STOP_FOR_COST, never a complete reference. No score-based automatic training
decision is supplied. Old STQS/futility is not a control dependency.

Finalist seed confirmation, material thresholds, total tuning resources and any
500K/full action require future prospective authority. Row bootstrap is
conditional on frozen weights and cannot estimate seed variance. One seed never
establishes robust superiority. A future FULL_CANDIDATE is not full-release or
official-role authority. Model-fidelity failures are IMPLEMENTATION_UNVERIFIED,
not evidence that the upstream architecture is scientifically negative.

## API / future integration map

| Stage | Read-only API | What it does not establish |
|---|---|---|
| P1 | `cross_scale_plan.DraftComparison.from_dict`, `review_draft_inputs` | No executable Spec, prospective RML or release; a pinned reference declaration still needs existing V5 artifact/runtime admission |
| P2 | `cross_scale_inputs.validate_members`, `validate_nested_cohorts`, `verify_sidecar`, `validate_graph_fixture` | Byte/fixture checks are not real cache acceptance or preprocessing authority |
| P3 | `cross_scale_checkpoint.ExactStepSnapshotSink`, `capture_checkpoint`, `evaluate_readonly` | No extra update, completed-pass invention, live selection or native qualification |
| P3/P5 | `cross_scale_analysis.import_accepted_trace`, `align_curves` | No inferred step, unique-members count, allocated cost, extrapolation or cross-role delta |
| P4 | `summarize_costs`, `allocation_view`, `profile_view` | No CPU-to-GPU conversion, overlapping window sum, missing-as-zero or unprofiled speed claim |
| P5/P6 | `cross_scale_backtest.reassessment_regression`, `historical_pair_class`, `backtest_decisions`; `interpret_run` | No activated STQS, quality-stop rule, robust single-seed win, automatic full candidate/release or RML outcome write |
| P7 | `model_fidelity.ModelFidelityEvidence` | Evidence shape/parity review only; no constructor, family registration or MODEL_FIDELITY_ACCEPTED certificate |

`ExactStepSnapshotSink` composes with the **existing checkpoint callback**, with
an existing atomic publisher supplied by its owner. The recipe's checkpoint
cadence must actually include the requested milestones; absent snapshots stay
`INCOMPLETE`. It does not change the loop/cadence, fabricate nearest-pass points
or automatically resume a diagnostic. Publication failure propagates through
the existing session's persistence-failure path. State hashes are semantic tensor
hashes, not claims of byte-identical Torch archive serialization.

The V1 evaluator supports **CPU synthetic authority only**, uses an independent
copy of the existing elementary model and the shared `evaluate_development`,
and verifies checkpoint/session tensors, buffers, optimizer/scheduler/EMA/RNG,
cursor and modes. It reads a caller-supplied checkpoint state: file transport,
original source validation and native model factories remain existing owners.
Live and raw EMA are supported; Clean-BN requires separate calibration evidence
and is deliberately rejected here. Future real evaluation must bind native
source/runtime, roles, loader and checkpoint acceptance under its owning action;
do not broaden the CPU permit into an execution switch.

The trace importer takes independently pinned trace, manifest and an owning
acceptance binding. It reuses canonical RML validators, preserves manifest
backtest eligibility and original metric names/units, and never grants scientific
acceptance from a status flag. The historical fixture embeds exact original JSON
bytes (inside escaped JSON strings), with source commit and SHA256 pins. Its
analysis-binding wrapper exists only in the temporary test repository; it is
not a manufactured historical acceptance or a new canonical RML record.

The draft has two scale recipes, exact diagnostic/primary step horizons, selected
weight state, seed list, model/source/feature/fidelity identities, native budget,
reference and independent role/history pins. Matched exposure shares the full
step LR trajectory and common-100K train transform. Tuned recipes declare their
own train-only transforms and a common per-system tuning ceiling. The future
compiler must validate all baseline/candidate registrations and budgets together;
this release intentionally contains no compiler.

Read-only output axes remain separate: `implementation_fidelity`,
`runtime_mechanical`, `scientific_status`, `scale_transfer`, `budget_decision`,
`replay`, `role_usage`, `promotion`. A two-seed favorable synthetic case returns
`EXPLORATORY_PRIMARY_POSITIVE`, never a robust ranking or automatic promotion.
Existing V5/RML validators own any later stronger claim and publication.
