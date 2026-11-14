# Experiment infrastructure quickstart

This page owns the end-to-end path for a new Kaggle T4x2 paired experiment, its
failure recovery, ordering and quota rules. The [CLI reference](EXPERIMENT_CLI.md)
owns individual flags. Routine work on an already declared, authorized run starts
with its platform skill and supplied contract/attempt records. A configuration or
successful preparation does not grant compute, protected roles, scientific
acceptance or production promotion.

## Happy path

Set these once per PowerShell session. Every command below runs from the
experiment worktree root. `platforms/local/run.py` runs the worktree's own code
with the primary checkout's interpreter; a plain `& $python -m molgap...` from a
worktree imports the primary checkout's code instead.

```powershell
$python = 'D:/文档/molgap/.venv/Scripts/python.exe'
$profilePath = 'D:/molgap-machine-profile.json'
$credentials = 'D:/下载/Key/.kaggle3_cli/kaggle.json'   # the account named in authority.account
$experiment = 'experiments/<experiment_id>/experiment.json'
$runId = '<logical_run_id>'; $attemptId = 'v1'
```

| # | Step | Command | Expected stdout `status` | Typical wall time |
|---|---|---|---|---|
| 1 | Choose the question from RML | `research_memory lookup/cards/gaps`, `catalog`, `check_quota.py` (below) | JSON / table | 4-7 min reading |
| 2 | Create the worktree | `git worktree add D:/molgap-worktrees/<q> -b codex/exp/<q> molgap-desktop` | - | 16 s |
| 3 | Write and commit inputs | `experiment.json`, `protocol.md`, `decision.md`, `STATUS.md` ([format](#supply-one-human-input)) | - | about 10 min |
| 4 | Prepare and submit | `submit_workflow.py --experiment ... --execute` | `SUBMISSION_BOUND` | 2-5 min |
| 5 | Commit generated records | `git add experiments/<id> research_memory/policies; git commit` | - | seconds |
| 6 | Wait | `retrieve_family_outputs.py --status ...` every 10 min | `OBSERVED`, `scheduler` = `running` then `complete` | kernel wall, about 1-2 h at 100K |
| 7 | Retrieve | `retrieve_family_outputs.py --all-arms --strict ... --execute` | `RETRIEVED`, prints `execution_root` | 1.5-2 min |
| 8 | Draft the closure | `draft-closure ...` | `DRAFT` | 7 s |
| 9 | Human decision | fill `outcome` and `decision_ref`, write `decision.md` | - | 2-3 min |
| 10 | Close | `accept_run.py --strict --closure-input ... --execute` | `COMPLETE`, `closure_complete: true` | about 1.5 min |
| 11 | Optional later recheck | `accept_run.py --strict` | `MECHANICALLY_VERIFIED`, `replay_readiness: VERIFIED`, `strict_mode: CANONICAL_READ_ONLY` | 10-14 s (up to about 40 s on a cold file cache) |
| 12 | Check RML and commit | `research_memory check --frozen`, update `STATUS.md`, commit | `RML derived outputs are frozen and current` | 25 s |

Targets: steps 3-4 (first input line to `SUBMISSION_BOUND`) under 10 minutes;
steps 7-11 under 5 minutes. A real run on 2026-11-01 measured step 4 at 2 min 49 s,
retrieval 89 s, draft 7 s and closure 92 s.

**Step 1: query the evidence.** Use RML evidence before proposing anything; do
not repeat a closed question. Pass `--repo-root .`: without the wrapper the
default is the primary checkout, which lacks unmerged experiment branches.

```powershell
& $python platforms/local/run.py --profile $profilePath --module molgap.research_memory -- --repo-root . lookup --mechanism <word>
& $python platforms/local/run.py --profile $profilePath --module molgap.research_memory -- --repo-root . cards --outcome NEGATIVE_UNDER_CONTRACT --term <word>
& $python platforms/local/run.py --profile $profilePath --module molgap.research_memory -- --repo-root . gaps
& $python platforms/local/run.py --profile $profilePath -- catalog
& $python platforms/kaggle/check_quota.py
```

`lookup` also accepts `--term`, `--family` and `--failure-mode`; field meanings are
in the [RML README](../../research_memory/README.md#query-analysis-and-corrections).
In `catalog`, an addon is usable by the compiler only when its `family_versions`
and `execution_modes` are non-empty for the preset's family (`neural_atom_k1`
version `3` for the only preset). Read the linked `decision_ref` files before
deciding. If no question would change a decision, submit nothing.

**Step 4: prepare and submit.** Under existing user authority for this account and
spend:

```powershell
& $python platforms/local/run.py --profile $profilePath --script platforms/kaggle/submit_workflow.py -- --experiment $experiment --repo-root . --credentials $credentials --execute
```

On success the output has `status: SUBMISSION_BOUND`, `kernel`, `kernel_id`,
`version`, `run_directory` and the `journal` path. `WAIT_SOURCE_READY` means rerun
the identical command. To check an addon on its real frozen state first, run the
same command without `--execute` and `--credentials` (`PREPARED_FOR_PLATFORM`),
then the sandbox, then the command with `--execute`, which reuses the preparation
(`reused_preparation: true`):

```powershell
$run = "D:/molgap-artifacts/runs/$runId/$attemptId"
& $python platforms/local/run.py --profile $profilePath -- sandbox --spec $run/inputs/experiment_spec.json --arm candidate --plan $run/inputs/workflow_plan.json --repo-root .
```

**Steps 6-7: wait and retrieve.** Each status call takes about 3 s, makes no
download and submits nothing. Keep each wait plus call under a 10 minute tool
limit.

```powershell
& $python platforms/local/run.py --profile $profilePath --script platforms/kaggle/retrieve_family_outputs.py -- --status --run-id $runId --attempt-id $attemptId --profile $profilePath --repo-root .
& $python platforms/local/run.py --profile $profilePath --script platforms/kaggle/retrieve_family_outputs.py -- --all-arms --strict --run-id $runId --attempt-id $attemptId --profile $profilePath --repo-root . --execute
```

Acceptance discovers the shared `_execution` directory beside retrieved arms.
Use `--execution-root` only when execution records are retained elsewhere.
An `--execute` call before the kernel ends returns `WAIT_REMOTE_TERMINAL` and
downloads nothing. If the scheduler shows `error` or `cancelled`, or the kernel
hit its wall cap (`STOP_FOR_COST`), see the [failure table](#failure-table).

**Steps 8-10: draft, decide, close.**

```powershell
$closureDir = 'experiments/<experiment_id>/closure_v1'
& $python platforms/local/run.py --profile $profilePath -- draft-closure --profile $profilePath --repo-root . --run-id $runId --attempt-id $attemptId --output-dir $closureDir
```

This writes `roles_<arm>.json`, `audit_map.json`, `measured_<run>.json`,
`closure_facts_<arm>.json` and `closure_input.draft.json`. In `measured_<run>.json`
and the candidate's `closure_facts`, `effect_meV` is candidate minus reference
development MAE in meV: negative means the candidate is better; `ci_meV` is a
row-bootstrap 95% interval. For more detail before deciding, optionally run
`accept_run.py --strict --diagnose` (14 s); it returns `BLOCKED` with
`explicit_outcome_and_decision_ref_required` and writes a post-hoc diagnostic
under `<run_dir>/reports/`.

Human decision points, never filled by a script:

- **`outcome` per arm.** Use one of `NO_TRAIN`, `POSITIVE_UNDER_CONTRACT`,
  `POSITIVE_BELOW_GATE`, `NEGATIVE_UNDER_CONTRACT`, `INCONCLUSIVE`,
  `STOP_FOR_COST`, `DUPLICATE_EVIDENCE`, `INFRASTRUCTURE_ONLY` or `CLOSED`. The
  usual pair is `CLOSED` for the reference arm and the result under the frozen
  `protocol.md` gate for the candidate. A closure-facts effect is accepted only
  when the reference arm closes `CLOSED`. `READY_FOR_DESKTOP` is blocked here.
- **`decision_ref` per arm**, a repository-relative path (usually
  `experiments/<id>/decision.md`). Write the final decision before closing;
  do not edit it after closure (see the erratum row below).
- **Protected-role states**, only if the draft left them `unknown` (see below).
- Optional `failure_mode` (one of `infra_failure`, `gate_or_cost_stop`,
  `lost_at_scale`, `early_gain_fades`, `endpoint_regression`, `below_gate`,
  `confounded_change`, `comparator_missing`, `discriminator_missing`,
  `boundary_pass`) and up to 8 `search_terms` in each `closure_facts_<arm>.json`.
  Leave the measured numbers unchanged.

Copy `closure_input.draft.json` to `closure_input.json` in the same directory,
fill each arm's `outcome` and `decision_ref`, and close:

```powershell
& $python platforms/local/accept_run.py --profile $profilePath --run-id $runId --attempt-id $attemptId --strict --closure-input $closureDir/closure_input.json --execute
```

The draft's `role_audit` is an absolute path, which works; a repository-relative
path keeps the committed file portable. The [CLI reference](EXPERIMENT_CLI.md#run-keyed-acceptance)
owns the closure file shape and path rules.

**Steps 11-12: record.** Strict closure already checks replay. Use
`accept_run.py --strict` only for later inspection or recovery, not after success.

```powershell
& $python platforms/local/run.py --profile $profilePath --module molgap.research_memory -- --repo-root . check --frozen
```

Closure already updated derived RML once; do not run `rebuild` after it. Write
the terminal attribution beside the decision (AGENTS.md "Terminal Attribution"),
update `STATUS.md` and commit the experiment directory and `research_memory/`
changes on the experiment branch.

## Failure table

Each status or message below was checked in `src/` or `platforms/`. Submitter
errors print `status`, `error_code` (`<STAGE>_<EXCEPTIONTYPE>`), `field`,
`reason` and `next_action`; `--verbose` does not expand them. On a non-error
result, `--verbose` prints the full owner report.

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
| accept | blocker `native_proof_dependencies_missing` | The run was retrieved without `--strict`. | Rerun step 7 with `--strict`. |
| accept | `ERROR`, message `Closure facts effect must be a measured pair ... against a reference arm closing CLOSED` | The candidate's effect needs a reference arm closing `CLOSED`. | Give the reference arm `CLOSED`, or set `effect_meV`, `ci_meV` and `source` to `null`. |
| accept | `ERROR`, message `Receipt retry conflict; refusing overwrite` from `draft-closure` | Draft files are immutable; a rerun produced different bytes. | Use a new `--output-dir` (`closure_v2`). |
| recheck | `BLOCKED`, replay `error: bound artifact SHA mismatch: <decision file>`, blockers `accepted_replay_evidence_missing_or_invalid`, `strict_replay_pool_membership_missing` | A decision file changed after closure. An erratum excuses it only in `research_memory validate`; strict recheck ignores errata by design (`pcqm-k1-init-noise-100k-s43`). | None. Keep the run as weak evidence; do not rewrite or re-close it. Prevent this by never editing a decision file after closure. |

### Paused STOP_FOR_COST runs

A run that reaches its frozen `allocation_wall_limit_seconds` stops with
`STOP_FOR_COST` and keeps complete-epoch resume state instead of an output manifest.
Retrieve its pause files, which records digests at download only:

```powershell
& $python platforms/local/run.py --profile $profilePath --script platforms/kaggle/retrieve_family_outputs.py -- --all-arms --paused --run-id $runId --attempt-id $attemptId --profile $profilePath --repo-root . --execute
```

The output is `PAUSED_RETRIEVED`, with files under
`D:/molgap-artifacts/retrieved/<run>/<attempt>/version<N>/<arm>`. This is not an
acceptance. Continuing needs new explicit user authorization for the extra quota
(a human decision). Then use `build-resume` per arm with that directory as
`--source-output`, `prepare-resume`, and the submitter's continuation arguments
(`--prepared`, `--recovery-authority`, `--recovery-authority-sha256`,
`--source-version N+1`, `--kernel-version N+1`), as described in the
[CLI reference](EXPERIMENT_CLI.md#local-lifecycle-by-run-id) and the
[submission adapter](KAGGLE_WORKFLOW_SUBMISSION.md#platform-operation). The
original wall limit cannot increase. Run the continuation from the run's original
experiment worktree; other checkouts fail the frozen release check. After the successor finishes, pass the
successor `--continuation-receipt` and `--resume-prepared` to retrieval and
`draft-closure`, and add `--resume-authority` at acceptance.

## Freeze and commit order

- The first `freeze-input` or `submit_workflow.py --experiment` call binds the
  current `HEAD` as the run's `source_commit`, under
  `D:/molgap-artifacts/runs/<logical_run_id>/<attempt_id>/inputs/`.
- Commit every human input (`experiment.json`, `protocol.md`, `decision.md`,
  `STATUS.md`, any `authority_ref` file) before that call. A modified tracked input
  fails with `V4 bundle source files must be committed before packaging`; an
  uncommitted new one with `Source absent from the frozen commit`.
- Make no commit between the freeze and `PREPARED_FOR_PLATFORM`. A commit in
  between needs a new `logical_run_id`. Running `submit_workflow.py` directly,
  without a separate `freeze-input`, keeps that window inside one command.
- Compilation writes untracked records: `experiments/<id>/generated/<hash>/`,
  `experiments/<id>/runs/...` and, for an unregistered policy,
  `research_memory/policies/<policy_id>.1.json`. Leave them untracked until
  `SUBMISSION_BOUND`, then review and commit them (step 5).
- An admission failure (`stage: workflow_readiness`) does not freeze, so the same
  run ID can be reused after a fix. Any change to a frozen `experiment.json` needs
  a new `logical_run_id`.
- A new attempt (`v2`) of the same run needs `retry_of` with matching authority and
  a reconciled terminal disposition; prefer a new `logical_run_id` for a corrected
  input that never submitted.

## Kaggle quota

- Kaggle charges the wall hours of the whole 2xT4 session, not device-hours: a
  25,024 s session used 6.95 h of quota.
- Quota resets weekly; `check_quota.py` prints the reset time (currently Saturday
  09:00 JST, 00:00 UTC). Weekly totals are currently 45 h on kaggle1 and 30 h on
  kaggle2 and kaggle3; read the live table.
- `& $python platforms/kaggle/check_quota.py` reads keys from `D:/下载/Key`
  (override with `--key-dir` or `MOLGAP_KAGGLE_KEY_DIR`; `--credentials
  label=path` for one file; `--json` for machine output). It is read-only, takes
  about 1.5 s and exits 1 if any account fails. Its `Username` column is the value
  for `authority.account`.
- Before dataset publication and again before the kernel POST, the submitter
  requires remaining quota of at least `min(allocated_t4_hours_ceiling,
  allocation_wall_limit_seconds / 3600)` and saves `quota_before_publication.json`
  and `quota_before_submit.json` in the run's `operations/`. Continuations compare
  against the wall limit alone.
- Set `allocation_wall_limit_seconds` to bound both the session and the quota
  draw. When it is hit the run pauses (`STOP_FOR_COST`) instead of losing its work.

## Select code and storage

The desktop machine profile is `D:/molgap-machine-profile.json`, outside Git:

```json
{
  "format": "molgap-machine-paths-v1",
  "python": "D:/文档/molgap/.venv/Scripts/python.exe",
  "data_root": "D:/molgap-data",
  "artifact_root": "D:/molgap-artifacts",
  "scratch_root": "D:/molgap-scratch"
}
```

Use the actual profile on each machine. The three storage roots must be separate
and outside the checkout. This profile does not download, copy or relocate old
data/checkpoints; their frozen manifests remain authoritative. A temporary
worktree reuses the primary dependency interpreter without reinstalling it.

The local wrapper `platforms/local/run.py --profile $profilePath [--module M |
--script S] -- ARGS` selects this checkout in an isolated child and reports its
interpreter and MolGap module on stderr. Without `--module` or `--script` it runs
`molgap.experiment_cli`. Do not redirect the shared editable install. With a
machine profile, each child owns a temporary subdirectory of `scratch_root`;
normal success, nonzero exit and launch exceptions clean that subdirectory. Keep
durable outputs and diagnostic logs in `artifact_root`. Forced process
termination can still leave scratch behind.

## Change the scientific delta

- For shared K1/3 and GPTrans/2, the scientific input compiler accepts typed
  learning-rate/weight-decay changes and catalog addons. Existing builders still
  own other recipe changes (for example a different pass count), full-training
  contracts and platform admission.
- Add an ordinary addon once in `experiment_catalog.py`, with its typed fields,
  supported family/version and execution modes/hooks. `experiment_execution.py`
  derives its read-only adapters from that catalog; do not register it twice.
  Shared recipe versions, modes and output profiles follow those adapters.
  Multiple versions for the same shared recipe owner require an explicit routing
  decision; the projection does not silently choose the newest version.
- Keep input ABI, objective and parameter ownership explicit. Reuse the family
  trainer and [runtime extension API](MODULAR_EXPERIMENT_RUNTIME.md). Use the
  [operation map](EXPERIMENT_ADDON_GUIDE.md#pick-the-operation) to find the owner.
  The [new addon checklist](EXPERIMENT_ADDON_GUIDE.md#new-addon-checklist) lists
  the module, catalog fields and local CPU check for a slot addon.

Desktop's existing family/addon declarations, accelerated defaults, full/SCNet
paths and native training implementations are preserved. Catalog registration
does not establish native qualification.

## Supply one human input

For a K1 post-BN 100K pair you fill in only the experiment's own details:
question, arms, the catalog addon and its parameters, authority and budget. Put
the file at `experiments/<experiment_id>/experiment.json` with a non-empty
`protocol.md` and `decision.md` (a placeholder until closure) and a `STATUS.md`
beside it, and commit them (see [freeze and commit order](#freeze-and-commit-order)).

```json
{
  "format": "molgap-scientific-experiment-v1",
  "experiment_id": "pcqm_k1_my_question_100k",
  "logical_run_id": "pcqm-k1-my-question-100k-s42",
  "attempt_id": "v1",
  "machine_profile": "D:/molgap-machine-profile.json",
  "timestamp": "2026-10-29T00:00:00Z",
  "question": { "policy_id": "pcqm-k1-my-question-100k", "action": "RUN_PAIRED_100K", "question": "...",
    "deficiency": "...", "changed_mechanism": "...", "supporting_evidence": ["<RML evidence id>"],
    "alternatives": ["..."], "closed_families": ["..."],
    "positive_decision": "...", "negative_decision": "..." },
  "profile": { "owner": "desktop", "track": "B", "platform": "kaggle",
    "hardware": "Tesla T4x2 FP32, one arm per device",
    "cheapest_falsifier": "...", "estimated_device_hours": 3.7, "estimated_wall_hours": 1.85 },
  "authority": { "format": "molgap-execution-authority-v1", "user_authorized": true,
    "account": "<kaggle username>", "arms": ["baseline", "candidate"], "seeds": [42],
    "same_run_reference_arm_id": "baseline", "allocated_t4_hours_ceiling": 8,
    "allocation_wall_limit_seconds": 14400, "cleanup_reserve_seconds": 180,
    "authority": "<who authorized what>", "scope": "<exact scope>",
    "development_role_label_use": true, "protected_roles_authorized": [] },
  "seeds": [42],
  "shared": {
    "preset_ref": { "path": "experiments/presets/k1-post-bn-100k-v1.json",
                    "sha256": "ea35cceb32bcc951d113f8265d9feedb4e0a31d5868a992e6e9eaac49f0fbc61" },
    "initial_state_ref": "D:/molgap-artifacts/initializations/structural-bias-norm-100k/k1.pt",
    "training": {}, "purpose": "single_module_ablation" },
  "arms": [ { "arm_id": "baseline" },
            { "arm_id": "candidate", "model": { "addons": [ { "name": "<catalog addon>", "config": {} } ] } } ],
  "same_run": { "reference_arm_id": "baseline" }
}
```

The preset pins the post-BN recipe, data declaration, target manifest,
initialization digest and model family. It is the only preset today; a GPTrans
or other-recipe question must supply those fields explicitly. The compiler
derives the rest from the experiment directory: `monitor_paths` (`STATUS.md`),
`role_ref` (`protocol.md`), `decision_ref` (`decision.md`), the Kaggle block
(kernel `<account>/molgap-<stem>-s42`, title and source dataset from the authority
`account` and the experiment ID, plus the fixed read-only
`<account>/pcqm4mv2-ogb-fixed-100k-v1` mount) and, when `policy_id` is not
registered and equals the experiment ID with dashes, a candidate
`research_memory/policies/<policy_id>.1.json` whose source digest is the
`protocol.md` digest. Any of these fields written explicitly in the input wins.
The question, arms, addon, authority, scope and budget have no default and are
never invented.

`supporting_evidence` takes RML evidence IDs only (the `evidence_id` of a
`v5_evidence.json`, listed as `result_evidence_ids` in
`research_memory/derived/trajectory_index.json`); cite papers in `protocol.md`.
`authority.account` must be the Kaggle username of `$credentials`; retrieve/status pick the file from profile `kaggle_credentials` by that account.

Training edits: `learning_rate`, `weight_decay`, `no_decay_modules`, `ema_decay`, `auxiliary_ema_decays`.
K1 reference/composed supports `optimizer: "k1-muon-adamw-fp32-v1"`, `"k1-muon-batched-adamw-fp32-v1"`, or `"k1-soap-adamw-fp32-v1"`.
All use hidden matrices and native AdamW remainder; batched Muon is not bit-exact with original Muon; SOAP first initializes preconditioning only.
Omit optimizer for unchanged AdamW; use `purpose: "optimizer_comparison"` for a pair.
Retained `reference_ref` takes `bundle_ref` plus `comparison_plan`. Default platform:
Kaggle T4x2, seed 42, fixed 100K mount; other mounts need explicit `platform`.

## Preparation details

Run files live at `<artifact_root>/runs/<logical_run_id>/<attempt_id>/`; `inputs`,
`prepared` and `operations` retain their distinct roles. `runtime-v1` packages
tracked runtime Python plus explicit scientific records. Source archives use a
content cache; each run still has its own Spec, package identity, release and
receipts. No data or weight download is implied by preparing a source package.

Both preparation entries resolve relative initialization paths against the selected
`repo_root`, independent of the caller's working directory. Within one invocation
from preparation through submission, identical package bytes reuse full validation;
each boundary still reads and hashes all six package files. Independent calls
start fresh. No verification result is persisted between preparations.
Re-reading an already prepared run does not create another timing observation file.

Continuation resolves the original frozen run before compiling new input. The
same input reuses its Spec/package/journal across unrelated HEAD changes once
prepared. Incomplete unpublished preparation can regenerate only from its bound
source commit; uncertain effects or journals remain `RECONCILIATION_REQUIRED`.
Never clear a journal to force a retry. Old `--spec` and `--prepared` APIs remain
available. IMS/SCNet retain their existing adapters and contracts.

## Local single-GPU route

For an explicitly authorized single-model local GPU exercise, use the existing
`prepare-family-training` entry with a local Spec, then run its frozen output:

```powershell
& $python platforms/local/train.py --prepared $prepared --repo-root . --data-root D:/molgap-data --output $output --pause-after-step 80
& $python platforms/local/train.py --prepared $prepared --repo-root . --data-root D:/molgap-data --output $output
```

The first process pauses after the specified absolute update; the next process
qualifies and restores that checkpoint under the original training horizon.
This adapter records the actual host/process, uses the registered native
preflight and trainer, and supports one arm on one device. It does not submit a
remote job. After completion, the same entry with `--accept-only --outcome
INFRASTRUCTURE_ONLY --decision-ref <decision.md> --role-audit <audit.json>
--execute` uses mechanical inspection, terminal staging and the existing RML
finalizer. `--closure-input FILE` (a single arm, outcome `INFRASTRUCTURE_ONLY`)
replaces the three terminal flags, and `--draft-role-audit <repo-dir>` drafts the
audit from the retained artifacts (`--protected-role-use ROLE=STATE` only from
observed facts; unknown blocks closure). The drafted audit is published
immutably, so a rerun with different role states needs a new directory.
The run-keyed `accept_run.py`, `draft-role-audit`, `draft-closure` and `diagnose`
need an `inputs/input_binding.json` that a local run does not have, so use these
flags here. This route does not establish scientific comparison or replay
eligibility.

## Test the changed flows once

Use the existing tests through one selected-checkout invocation. Select only the
affected flows; overlapping selections are deduplicated before pytest starts.

```powershell
& $python platforms/local/run.py --profile $profilePath --module molgap.local_testing -- --scope package --scope prepare -- -q
```

Available scopes: `runtime`, `package`, `prepare`, `train`, `submit`, `accept`
and `resume`. The [verification conventions](INFRASTRUCTURE_VERIFICATION.md)
own their real execution boundaries and supported claims. `--list` shows fixed
pytest node IDs; adding a test to the same file cannot expand the selection.
Direct pytest remains available for affected lower-level invariants. Selection
filters/skips mean partial coverage; these journeys are not a mandatory release
gate or native qualification.

Run relevant tests after the final code changes. Re-run a failing or affected
selection after a fix; do not repeat successful unchanged selections at every
handoff, commit or merge. The profile wrapper removes its own test scratch on
exit; save any needed pytest logs/JUnit reports under `artifact_root` before
returning. Do not create another worktree or environment merely for a test stage.

## Distribute tests to server

Git synchronization carries the tracked tests and configuration. For an extracted
source distribution, use the existing source-bundle writer through the test entry:

```powershell
& $python platforms/local/run.py --profile $profilePath --module molgap.local_testing -- --repo-root . --package D:/molgap-artifacts/test-source
```

Commit source changes first. The output directory must be absent or empty.
`--list --package <output>` previews the inventory without building it. The bundle
includes active package Python, tracked test Python, pytest configuration and
compact support inputs for the listed infrastructure scopes. It excludes weights,
datasets, accounts and dependency environments.

Transfer the bundle and its hash/inventory sidecars, then extract
`source.tar.gz` into an empty directory. On the server, use its dependency
environment with pytest and Git installed and run from that directory:

```bash
python platforms/local/run.py --module molgap.local_testing -- --scope runtime -- -q
```

The [desktop migration record](DESKTOP_INFRASTRUCTURE_UPGRADE.md) owns provenance
and executed verification. [Storage retirement](LOCAL_STORAGE_RETIREMENT.md)
owns payload and completed-worktree retirement. After a run closes, its
`--report` shows where `artifact_root` bytes sit. `--dedupe` is also read-only;
hardlink mutation is disabled because resume writers may overwrite shared bytes.
