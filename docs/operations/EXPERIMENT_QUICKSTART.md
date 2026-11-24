# Experiment infrastructure quickstart

This page owns the Kaggle T4x2 paired-experiment path, failure recovery, ordering
and quota rules. The [lifecycle map](LIFECYCLE_MAP.md) owns the command order; the
[CLI reference](EXPERIMENT_CLI.md) owns flags. Preparation grants no compute,
protected role, acceptance or promotion.

## Happy path

Set these once per PowerShell session, in the experiment worktree root. Use
`platforms/local/run.py` (worktree code, primary interpreter), not a plain `-m molgap`.

```powershell
$python = 'D:/文档/molgap/.venv/Scripts/python.exe'
$profilePath = 'D:/molgap-machine-profile.json'
$credentials = 'D:/下载/Key/.kaggle3_cli/kaggle.json'   # the account named in authority.account
$experiment = 'experiments/<experiment_id>/experiment.json'
$runId = '<logical_run_id>'; $attemptId = 'v1'
$rml = @('platforms/local/run.py','--profile',$profilePath,'--module','molgap.research_memory','--','--repo-root','.')
```

Query RML first (`& $python @rml lookup --mechanism <word>`; also `cards`, `levers`,
`gaps`) and always pass `--repo-root .`, or you read the primary checkout. Create the
worktree with `git worktree add D:/molgap-worktrees/<q> -b codex/exp/<q> molgap-desktop`,
commit the inputs ([format](#supply-one-human-input)), then follow the
[lifecycle map](LIFECYCLE_MAP.md#commands).

**Submit** (under existing user authority for the account and spend). Success
prints `SUBMISSION_BOUND` and the `journal` path; `WAIT_SOURCE_READY` means rerun the
identical command. To test an addon on its frozen state first, run it without
`--execute` and `--credentials` (`PREPARED_FOR_PLATFORM`), then the `sandbox` command with
`--spec <run>/inputs/experiment_spec.json --arm candidate --plan <run>/inputs/workflow_plan.json`.

```powershell
& $python platforms/local/run.py --profile $profilePath --script platforms/kaggle/submit_workflow.py -- --experiment $experiment --repo-root . --credentials $credentials --execute
```

**Wait and retrieve.** `check_logs.py --run-id $runId --tail 5` reads live progress;
`retrieve_family_outputs.py --status` checks terminal state; the same script with
`--all-arms --strict --execute` downloads (`RETRIEVED`, prints `execution_root`).
`--execute` before the kernel ends returns `WAIT_REMOTE_TERMINAL`. Acceptance finds
the shared `_execution` directory beside the arms (`--execution-root` overrides).

**Draft, decide, close.** `draft-closure --output-dir experiments/<id>/closure_v1`
writes the role audit, measured numbers, per-arm facts and `closure_input.draft.json`.
`effect_meV` is candidate minus reference development MAE (negative is better); `ci_meV`
is a row-bootstrap 95% interval. Copy the draft to `closure_input.json`, fill each arm's
`outcome` and `decision_ref` (a repository-relative path; never edit the decision after
closing), then run `accept_run.py --strict --closure-input ... --execute` (`COMPLETE`). The usual pair is `CLOSED` for the
reference arm and the gate result for the candidate; an effect counts only against
a reference closing `CLOSED`. `READY_FOR_DESKTOP` is blocked here. Optional
`failure_mode` and up to 8 `search_terms` go in each facts file (see the
[closure format](RML_INTERFACES.md#decisionfacts-molgap-closure-facts-v1)); leave
measured numbers unchanged. Closure already updated derived RML: do not run `rebuild`.
Write the attribution beside the decision, update `STATUS.md`, run
`& $python @rml check --frozen` and commit on the experiment branch.

## Failure table

Submitter errors print `status`, `error_code`, `field`, `reason` and `next_action`;
`--verbose` adds the owner report only on a non-error result.

| Where | Output | Cause | Next command |
|---|---|---|---|
| submit | `RECONCILIATION_REQUIRED`, reason `Unpublished preparation requires the original bound source checkout; current HEAD is not substituted` | A commit landed between the freeze and a successful preparation, for example committing the generated policy after `freeze-input` ([rule](#freeze-and-commit-order)). | Set a new `logical_run_id` (`...-r2`) in `experiment.json`, commit, rerun step 4. Do not check out the old commit. |
| submit | `BLOCKED`, `LOCAL_PREPARATION_VALUEERROR`, `field` and `reason` (for example `Local validation failed: Unknown addon/version`) | The input fails a local check; `reason` holds the message. `Unknown addon/version` means the addon is not in this checkout's catalog. | Fix the named field and commit. If `D:/molgap-artifacts/runs/<run>/<attempt>/inputs/` already exists, also set a new `logical_run_id`. |
| submit | `BLOCKED`, `field: logical_run_id`, reason `Scientific input changed; create an explicit new run rather than overwrite this run` | `experiment.json` changed after its run was frozen. | New `logical_run_id`, commit, rerun step 4. |
| submit | `BLOCKED`, `stage: workflow_readiness`, `blockers` list | Admission failed before the freeze; nothing was frozen. | Fix the listed blockers, commit, rerun with the same run ID. |
| submit | `BLOCKED`, `LOCAL_PREPARATION_FILENOTFOUNDERROR`, reason `A required local file is missing` | A referenced file (preset, initialization, `protocol.md`, `decision.md`) is absent. | Create or correct the path, commit, rerun. |
| submit | `RECONCILIATION_REQUIRED`, reason `Published/unknown preparation or platform effects require reconciliation` | A previous preparation published prospective records or left an unknown state. | Inspect `prepared/workflow_report.json`; if it published, finish with `submit_workflow.py --recover-prepared --prepared <run>/prepared` ([recovery](KAGGLE_WORKFLOW_SUBMISSION.md#local-admission)), otherwise use a new `logical_run_id`. |
| submit | `WAIT_SOURCE_READY` (exit 1) | The source dataset was not ready within the 300 s poll. | Rerun the identical command; the journal resumes. |
| submit | `RECONCILIATION_REQUIRED` with `phase` `SOURCE_UNKNOWN`, `SOURCE_PUBLISHED` or `SUBMIT_UNKNOWN` | An uncertain dataset create, a persistent 403 or a lost kernel response. | Stop. Inspect `operations/submission_state.json` and the Kaggle kernel page; never clear the journal or change the slug ([submission adapter](KAGGLE_WORKFLOW_SUBMISSION.md#platform-operation)). |
| submit | `BLOCKED`, `error_code: PLATFORM_VALUEERROR`, `field: platform.journal` | A platform-stage check failed, for example quota below the allocation ceiling, kernel already exists or credential owner differs. | Run `check_quota.py`; confirm `$credentials` belongs to `authority.account`; inspect the journal. |
| retrieve | `READY_FOR_RETRIEVAL` with `pin_command` | Run without `--execute`; manifests not yet pinned. | Run step 7 with `--execute`; credentials resolve from profile `kaggle_credentials[<receipt account>]`. |
| retrieve | `WAIT_REMOTE_TERMINAL` | The kernel version is still running. | Wait; poll with `--status`. |
| retrieve | `BLOCKED`, `stage: remote_manifests` | The kernel stopped with no `output_manifest.json`, usually a `STOP_FOR_COST` pause at its wall cap. | Retrieve unstrict with `--all-arms --paused --execute` and follow [paused runs](#paused-stop_for_cost-runs). |
| retrieve | `BLOCKED`, error `ValueError`, with `--paused --strict` | A paused run has no strict evidence. | Drop `--strict`. |
| accept | `STAGED_DRAFT` (without `--strict`) or `BLOCKED` (with `--strict`), blockers `<arm>: explicit_outcome_and_decision_ref_required` | No outcome or decision was supplied. Expected before step 10. | Steps 8-10. |
| accept | blocker `<arm>: observed_protected_role_audit_missing` | No role audit was supplied. | Steps 8-10; `--closure-input` carries `role_audit`. |
| accept | blocker `observed_protected_role_audit_incomplete` | The audit leaves a protected role `unknown`: no `data_open_provenance.json` and no structural evidence. | Rerun `draft-closure` into a new `--output-dir` with `--protected-role-use ROLE=STATE` only from observed facts, or leave the run unclosed ([role drafting](EXPERIMENT_CLI.md#role-audits)). |
| accept | blocker `positive_or_ready_requires_owning_scientific_acceptance` | A positive outcome without `--strict`, or `READY_FOR_DESKTOP`. | Close positives with `--strict`; `READY_FOR_DESKTOP` has its own qualification. |
| accept | blocker `native_proof_dependencies_missing` | The run was retrieved without `--strict`. | Rerun step 7 with `--strict`. Exception: if the protocol pre-registered a non-strict path (for example "decision 4.1(b)", used when `runtime_certificate_scope` is the only `SCIENTIFIC_CONTRACT_MISMATCH` field), keep the non-strict retrieval and accept without `--strict`; see the Kaggle skill section "Pre-registered non-strict path". |
| accept | `ERROR`, message `Closure facts effect must be a measured pair ... against a reference arm closing CLOSED` | The candidate's effect needs a reference arm closing `CLOSED`. | Give the reference arm `CLOSED`, or set `effect_meV`, `ci_meV` and `source` to `null`. |
| accept | `ERROR`, message `Receipt retry conflict; refusing overwrite` from `draft-closure` | Draft files are immutable; a rerun produced different bytes. | Use a new `--output-dir` (`closure_v2`). |
| recheck | `BLOCKED`, replay `error: bound artifact SHA mismatch: <decision file>`, blockers `accepted_replay_evidence_missing_or_invalid`, `strict_replay_pool_membership_missing` | A decision file changed after closure. An erratum excuses it only in `research_memory validate`; strict recheck ignores errata by design (`pcqm-k1-init-noise-100k-s43`). | None. Keep the run as weak evidence; do not rewrite or re-close it. Prevent this by never editing a decision file after closure. |

### Paused STOP_FOR_COST runs

A run that hits `allocation_wall_limit_seconds` pauses with `STOP_FOR_COST` and keeps
resume state. Retrieve it with `retrieve_family_outputs.py --all-arms --paused --execute`
(`paused_root`). After explicit authorization for extra quota, run `prepare-resume
--paused-root`, then submit with `--prepared` and `--recovery-authority`
([CLI](EXPERIMENT_CLI.md#local-lifecycle-by-run-id), [adapter](KAGGLE_WORKFLOW_SUBMISSION.md#platform-operation)).
Use the original worktree; pass the successor directory as `--resume-prepared` to
retrieval, `draft-closure` and acceptance.

## Freeze and commit order

- The first `freeze-input` or `submit_workflow.py --experiment` call binds the current
  `HEAD` as `source_commit` under `<artifact_root>/runs/<logical_run_id>/<attempt_id>/inputs/`.
- Commit every human input first (`experiment.json`, `protocol.md`, `decision.md`,
  `STATUS.md`, `pre_submit_review.json`, any `authority_ref` file); otherwise packaging fails
  with `V4 bundle source files must be committed` or `Source absent from the frozen commit`.
- Make no commit between the freeze and `PREPARED_FOR_PLATFORM` (one needs a new
  `logical_run_id`). Commit generated records only after `SUBMISSION_BOUND`.
- An admission failure (`workflow_readiness`) freezes nothing, so the run ID can be reused.
  Any change to a frozen `experiment.json` needs a new `logical_run_id`; a new attempt
  (`v2`) needs `retry_of`, matching authority and a reconciled terminal state.

## Kaggle quota

Kaggle charges wall hours of the whole 2xT4 session and resets weekly.
`platforms/kaggle/check_quota.py` (read-only) prints the live table and the `Username` for
`authority.account`. The submitter needs quota of at least `min(allocated_t4_hours_ceiling,
allocation_wall_limit_seconds / 3600)` before publication and the POST.

## Storage, addons and tests

- The machine profile (`D:/molgap-machine-profile.json`, outside Git) names `python`,
  `data_root`, `artifact_root` and `scratch_root`; the roots are separate and outside the
  checkout. `platforms/local/run.py --profile P [--module M | --script S] -- ARGS` runs the
  selected checkout in a child with its own scratch directory; without a module or script it
  runs `molgap.experiment_cli`. Do not redirect the shared editable install.
- The compiler accepts typed learning-rate/weight-decay changes and catalog addons for K1/3 and
  GPTrans/2. Add an addon once in `experiment_catalog.py`; `experiment_execution.py` derives its
  adapters. The [operation map](EXPERIMENT_ADDON_GUIDE.md#pick-the-operation) and
  [checklist](EXPERIMENT_ADDON_GUIDE.md#new-addon-checklist) name the owners. Registration is
  not native qualification.
- Test the affected flows once: `platforms/local/run.py --profile $profilePath --module molgap.local_testing -- --scope package --scope prepare -- -q`
  (scopes `runtime`, `package`, `prepare`, `train`, `submit`, `accept`, `resume`; `--list` shows
  node IDs; `--repo-root . --package <empty dir>` writes a source bundle for another machine).
  The [verification conventions](INFRASTRUCTURE_VERIFICATION.md) own the boundaries.
- Local single-GPU exercise: `prepare-family-training`, then `platforms/local/train.py --prepared P --repo-root . --data-root D:/molgap-data --output O`.
  Close it with `--accept-only --closure-input FILE --execute` (one arm, `INFRASTRUCTURE_ONLY`).
  Run-keyed `accept_run.py` and `draft-closure` need `input_binding.json`, absent here.

## Supply one human input

Put `experiment.json` at `experiments/<experiment_id>/` with a non-empty `protocol.md` and
`decision.md` (a placeholder until closure), `STATUS.md` and `pre_submit_review.json`, and
commit them ([order](#freeze-and-commit-order)). Copy a real input such as
`experiments/pcqm_k1_tail_regularization_500k/experiment.json`. You supply the question, arms,
addon and parameters, authority and budget; none has a default. The compiler derives
`monitor_paths`, `role_ref`, `decision_ref`, the Kaggle block (standard fixed 100K or 500K
read-only mount) and, for an unregistered `policy_id`, a candidate policy file. Explicit
fields win. `supporting_evidence` takes RML evidence IDs only (`result_evidence_ids` in
`research_memory/derived/trajectory_index.json`); cite papers in `protocol.md`.
`authority.account` must be the username of `$credentials`. Training edits: `learning_rate`,
`weight_decay`, `no_decay_modules`, `ema_decay`, `auxiliary_ema_decays`, K1 `optimizer`
(`k1-muon-adamw-fp32-v1`, `k1-muon-batched-adamw-fp32-v1`, `k1-soap-adamw-fp32-v1`).
