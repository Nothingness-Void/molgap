# Experiment infrastructure quickstart

For an already declared, authorized run, start with its platform skill.
This is the entry for authoring new work. Read `AGENTS.md`, `CURRENT_STATE.md`, the
relevant `ROADMAP.md` section and the question's decision first. Query existing
RML before proposing a new experiment. This guide grants no compute, role,
scientific acceptance or cross-machine custody.

## Happy path

Set these once per PowerShell session. Every command below runs from the
experiment worktree root. The `.venv` editable install points at the primary
checkout, so run scripts through `platforms/local/run.py`, which runs the worktree's
code; a plain `& $python -m molgap...` imports the primary checkout's code instead.

```powershell
$python = 'C:/Users/Adminn/Documents/molgap/.venv/Scripts/python.exe'
$profilePath = 'E:/MolGap/machine-profile.json'
$credentials = '<path>'   # the account's key file, passed explicitly as --credentials; submit rejects one inside a Git checkout or whose username differs from launch.account
$experiment = 'experiments/<experiment_id>/experiment.json'
$runId = '<logical_run_id>'; $attemptId = 'v1'
```

| # | Step | Command | Expected stdout `status` | Typical wall time |
|---|---|---|---|---|
| 1 | Choose the question from RML | `research_memory lookup/cards/gaps`, `catalog`, `check_quota.py` (below) | JSON / table | 4-7 min reading |
| 2 | Create the worktree | `git worktree add` a dedicated worktree of `origin/molgap-server`; do not share one checkout, because prepare stops if a packaged source file is uncommitted or `HEAD` moves | - | 16 s |
| 3 | Write and commit inputs | `experiment.json`, `protocol.md`, `decision.md`, `STATUS.md` ([format](#3-write-one-scientific-input)) | - | about 10 min |
| 4 | Prepare and submit | `submit_workflow.py --experiment ... --execute` | `SUBMISSION_BOUND` | 2-5 min |
| 5 | Commit generated records | `git add experiments/<id> research_memory/policies; git commit` | - | seconds |
| 6 | Wait | `retrieve_family_outputs.py --status ...` every 10 min | `OBSERVED`, `scheduler` = `running` then `complete` | kernel wall, about 1-2 h at 100K |
| 7 | Retrieve | `retrieve_family_outputs.py --all-arms --strict ... --execute` | `RETRIEVED`, prints `execution_root` | 1.5-2 min |
| 8 | Draft the closure | `draft-closure ...` | `DRAFT` | 7 s |
| 9 | Human decision | fill `outcome` and `decision_ref`, write `decision.md` | - | 2-3 min |
| 10 | Close | `accept_run.py --strict --closure-input ... --execution-root ... --execute` | `COMPLETE`, `closure_complete: true` | about 1.5 min |
| 11 | Strict read-only recheck | `accept_run.py --strict` | `MECHANICALLY_VERIFIED`, `replay_readiness: VERIFIED`, `strict_mode: CANONICAL_READ_ONLY` | 10-14 s (up to about 40 s on a cold file cache) |
| 12 | Check RML and commit | `research_memory check --frozen`, update `STATUS.md`, commit | `RML derived outputs are frozen and current` | 25 s |

Targets: steps 3-4 (first input line to `SUBMISSION_BOUND`) under 10 minutes;
steps 7-11 under 5 minutes. A Desktop run on 2026-11-01 measured step 4 at 2 min 49 s,
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
$run = "E:/MolGap/artifacts/runs/$runId/$attemptId"
& $python platforms/local/run.py --profile $profilePath -- sandbox --spec $run/inputs/experiment_spec.json --arm candidate --plan $run/inputs/workflow_plan.json --repo-root .
```

**Steps 6-7: wait and retrieve.** Each status call takes about 3 s, makes no
download and submits nothing. Keep each wait plus call under a 10 minute tool
limit.

```powershell
& $python platforms/local/run.py --profile $profilePath --script platforms/kaggle/retrieve_family_outputs.py -- --status --run-id $runId --attempt-id $attemptId --profile $profilePath --repo-root . --credentials $credentials
& $python platforms/local/run.py --profile $profilePath --script platforms/kaggle/retrieve_family_outputs.py -- --all-arms --strict --run-id $runId --attempt-id $attemptId --profile $profilePath --repo-root . --credentials $credentials --execute
```

Keep `execution_root` from the retrieval output; it is
`E:/MolGap/artifacts/retrieved/<run>/<attempt>/version<N>/_execution`.
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
& $python platforms/local/accept_run.py --profile $profilePath --run-id $runId --attempt-id $attemptId --strict --closure-input $closureDir/closure_input.json --execution-root <execution_root> --execute
```

The draft's `role_audit` is an absolute path, which works; a repository-relative
path keeps the committed file portable. The [CLI reference](EXPERIMENT_CLI.md#local-lifecycle-by-run-id)
owns the closure file shape and path rules.

**Steps 11-12: recheck and record.**

```powershell
& $python platforms/local/accept_run.py --profile $profilePath --run-id $runId --attempt-id $attemptId --strict
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
| submit | `BLOCKED`, `LOCAL_PREPARATION_VALUEERROR`, `field` and `reason` (for example `Local validation failed: Unknown addon/version`) | The input fails a local check; `reason` holds the message. `Unknown addon/version` means the addon is not in this checkout's catalog; `Expected one uniquely mounted pinned manifest` means the pinned fixed-dataset manifest is absent or not unique under the profile `data_root` (for example `E:/MolGap/data/pcqm_fixed_100k_v1/manifest.json`). | Fix the named field and commit. If `E:/MolGap/artifacts/runs/<run>/<attempt>/inputs/` already exists, also set a new `logical_run_id`. |
| submit | `BLOCKED`, `field: logical_run_id`, reason `Scientific input changed; create an explicit new run rather than overwrite this run` | `experiment.json` changed after its run was frozen. | New `logical_run_id`, commit, rerun step 4. |
| submit | `BLOCKED`, `stage: workflow_readiness`, `blockers` list | Admission failed before the freeze; nothing was frozen. | Fix the listed blockers, commit, rerun with the same run ID. |
| submit | `BLOCKED`, `LOCAL_PREPARATION_FILENOTFOUNDERROR`, reason `A required local file is missing` | A referenced file (preset, initialization, `protocol.md`, `decision.md`) is absent. | Create or correct the path, commit, rerun. |
| submit | `RECONCILIATION_REQUIRED`, reason `Published/unknown preparation or platform effects require reconciliation` | A previous preparation published prospective records or left an unknown state. | Inspect `prepared/workflow_report.json`; if it published, finish with `submit_workflow.py --recover-prepared --prepared <run>/prepared` ([recovery](KAGGLE_WORKFLOW_SUBMISSION.md#local-admission)), otherwise use a new `logical_run_id`. |
| submit | `WAIT_SOURCE_READY` (exit 1) | The source dataset was not ready within the 300 s poll. | Rerun the identical command; the journal resumes. |
| submit | `RECONCILIATION_REQUIRED` with `phase` `SOURCE_UNKNOWN`, `SOURCE_PUBLISHED` or `SUBMIT_UNKNOWN` | An uncertain dataset create, a persistent 403 or a lost kernel response. | Stop. Inspect `operations/submission_state.json` and the Kaggle kernel page; never clear the journal or change the slug ([submission adapter](KAGGLE_WORKFLOW_SUBMISSION.md#platform-operation)). |
| submit | `BLOCKED`, `error_code: PLATFORM_VALUEERROR`, `field: platform.journal` | A platform-stage check failed, for example quota below the allocation ceiling, kernel already exists or credential owner differs. | Run `check_quota.py`; confirm `$credentials` belongs to `authority.account`; inspect the journal. |
| retrieve | `READY_FOR_RETRIEVAL` with `pin_command` | Run without `--execute`; manifests not yet pinned. | Run step 7 with `--credentials` and `--execute`. |
| retrieve | `WAIT_REMOTE_TERMINAL` | The kernel version is still running. | Wait; poll with `--status`. |
| retrieve | `BLOCKED`, `stage: remote_manifests` | The kernel stopped with no `output_manifest.json`, usually a `STOP_FOR_COST` pause at its wall cap. | Retrieve unstrict with `--all-arms --paused --execute` and follow [paused runs](#paused-stop_for_cost-runs). |
| retrieve | `BLOCKED`, error `ValueError`, with `--paused --strict` | A paused run has no strict evidence. | Drop `--strict`. |
| accept | `STAGED_DRAFT` (without `--strict`) or `BLOCKED` (with `--strict`), blockers `<arm>: explicit_outcome_and_decision_ref_required` | No outcome or decision was supplied. Expected before step 10. | Steps 8-10. |
| accept | blocker `<arm>: observed_protected_role_audit_missing` | No role audit was supplied. | Steps 8-10; `--closure-input` carries `role_audit`. |
| accept | blocker `observed_protected_role_audit_incomplete` | The audit leaves a protected role `unknown`: no `data_open_provenance.json` and no structural evidence. | Rerun `draft-closure` into a new `--output-dir` with `--protected-role-use ROLE=STATE` only from observed facts, or leave the run unclosed ([role drafting](EXPERIMENT_CLI.md#local-lifecycle-by-run-id)). |
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
& $python platforms/local/run.py --profile $profilePath --script platforms/kaggle/retrieve_family_outputs.py -- --all-arms --paused --run-id $runId --attempt-id $attemptId --profile $profilePath --repo-root . --credentials $credentials --execute
```

The output is `PAUSED_RETRIEVED`, with files under
`E:/MolGap/artifacts/retrieved/<run>/<attempt>/version<N>/<arm>`. This is not an
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
  `E:/MolGap/artifacts/runs/<logical_run_id>/<attempt_id>/inputs/`.
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
- `& $python platforms/kaggle/check_quota.py` reads keys from `MOLGAP_KAGGLE_KEY_DIR`
  (override with `--key-dir`; `--credentials
  label=path` for one file; `--json` for machine output). It is read-only, takes
  about 1.5 s and exits 1 if any account fails. Its `Username` column is the value
  for `authority.account`.
- Before dataset publication and again before the kernel POST, the submitter
  requires `min(allocated_t4_hours_ceiling, allocation_wall_limit_seconds / 3600)`
  and saves `quota_before_publication.json` and `quota_before_submit.json` in the
  run's `operations/`. Continuations compare against the wall limit alone.
- Set `allocation_wall_limit_seconds` to bound both the session and the quota
  draw. When it is hit the run pauses (`STOP_FOR_COST`) instead of losing its work.

## 1. Select code and local storage

Use the project dependency interpreter; do not repoint its shared editable
installation. `platforms/local/run.py` selects this checkout in an isolated child
and logs the interpreter and actual MolGap module. Configure a machine-local
profile once, outside Git:

```json
{
  "format": "molgap-machine-paths-v1",
  "python": "C:/Users/Adminn/Documents/molgap/.venv/Scripts/python.exe",
  "data_root": "E:/MolGap/data",
  "artifact_root": "E:/MolGap/artifacts",
  "scratch_root": "E:/MolGap/scratch"
}
```

Paths are machine locations, not evidence identities. Use the actual profile on
each host; do not copy another machine's absolute paths. Storage roots must be
separate and outside the selected checkout. Old data/checkpoints remain where
their retained manifests say; this profile does not rebuild or relocate them.

```powershell
$python = 'C:/Users/Adminn/Documents/molgap/.venv/Scripts/python.exe'
$profilePath = 'E:/MolGap/machine-profile.json'
& $python platforms/local/run.py --profile $profilePath -- catalog
```

The capability catalog is the authority for declarations and derived execution
support. `DECLARATION_ONLY` and `CPU_SYNTHETIC_ONLY` are real limits. Registered
support still needs its recipe/data/reference/runtime qualifications.

## 2. Change only the scientific delta

- Shared K1/3 and GPTrans/2: edit `learning_rate` / `weight_decay` in the scientific
  input below. The compiler calls the existing Recipe and declaration builders;
  batch, sampler, normalization, schedule, selection and budget remain frozen.
  Other recipe changes still need their owning typed builder, not arbitrary overrides.
- Ordinary addon: implement the module and tests; add **one** `AddonContract`
  in `experiment_catalog.py`, including typed fields, family versions and
  `execution_modes` plus the reviewed `apply_hook` or `policy_hook` when needed.
  `experiment_execution.py` derives its read-only adapters; do not register the
  same addon there again. See the [addon API](EXPERIMENT_ADDON_GUIDE.md).
- New family: declare `FamilyContract.execution` in that same catalog and supply
  the actual model/input ABI and focused checks. Reuse the existing training
  owner. The [graph extension API](GRAPH_SCREEN_EXTENSION.md) applies only to
  its narrow pure-2D PCQM Gap contract.

Keep composition order and parameter ownership explicit before optimizer
construction. New objectives or input structures need their owning step/input
adapter, not a disguised ordinary addon. K1 sparse and GPTrans dense contexts
remain different. Use the [runtime API](MODULAR_EXPERIMENT_RUNTIME.md) for slots,
Live/Raw/Clean actions and source-free single-batch sandbox checks.

Formal releases bind committed source and frozen scientific records. Sandbox
code may be uncommitted, but is not a formal release. Do not retrain an accepted
baseline or add seeds to fill an allocation.

## 3. Write one scientific input

To start from a frozen scientific run, generate an editable draft:

```powershell
& $python platforms/local/run.py --profile $profilePath -- derive-input --from-run $sourceRunDirectory --experiment-id $experimentId --run-id $newRunId --profile $profilePath --output $inputPath
```

Review the copied question and delta; supply fresh authority before preparation.
`build-preset --help` lists the retained files needed to export baseline pins
without typing hashes. Both commands write ordinary existing input formats.

PPGT/3 and ESA/4 also use this front door through the
[configured Modern2D contract](MODERN2D_CONFIGURED_WORKFLOW.md), including
ordinary readout addons, explicit pass horizons and accepted indexed roles.
Scientific addon screens require a frozen candidate/reference comparison;
execution diagnostics retain their explicit context-only scope. The linked
contract owns the one-time baseline bootstrap and automatic runtime audit.

For shared K1/3 and GPTrans/2, one human JSON supplies the scientific question,
supported parameter/addon delta, frozen recipe/data/reference inputs, seed 42,
existing authority, platform metadata and local paths. It does not require a
hand-written new Spec, workflow, prospective envelope or calculated hash:

```json
{
  "format": "molgap-scientific-experiment-v1",
  "experiment_id": "optimizer-question",
  "logical_run_id": "optimizer-question-screen42",
  "attempt_id": "v1",
  "machine_profile": "E:/MolGap/machine-profile.json",
  "authority_ref": "experiments/optimizer-question/authority.json",
  "monitor_paths": ["experiments/optimizer-question/STATUS.md"],
  "timestamp": "2026-10-16T00:00:00Z",
  "platform_ref": "experiments/optimizer-question/platform.json",
  "role_ref": "experiments/reference/roles.json",
  "decision_ref": "experiments/optimizer-question/decision.md",
  "seeds": [42],
  "question": {
    "policy_id": "existing-registered-policy", "action": "EXISTING_ALLOWED_ACTION",
    "question": "Does this optimizer change improve the frozen screen?",
    "deficiency": "State an evidence-backed deficiency",
    "changed_mechanism": "optimizer",
    "supporting_evidence": ["existing-evidence-id"],
    "alternatives": ["Run noise"], "closed_families": ["existing-closed-route"],
    "positive_decision": "Review terminal evidence", "negative_decision": "Retain reference",
    "reference_ids": ["existing-reference-evidence-id"], "prior_trajectories": []
  },
  "profile": {
    "owner": "server", "track": "B", "platform": "kaggle", "hardware": "T4x2",
    "cheapest_falsifier": "One bounded paired seed-42 screen",
    "estimated_device_hours": null, "estimated_wall_hours": null
  },
  "arms": [{
    "arm_id": "candidate",
    "model": {"family": "neural_atom_k1", "version": "3", "addons": []},
    "recipe_ref": "experiments/reference/contract.json",
    "data_ref": {"spec_ref": "experiments/reference/experiment_spec.json", "arm_id": "reference"},
    "initial_state_ref": "E:/MolGap/artifacts/frozen-initialization.pt",
    "training": {"learning_rate": 0.0005},
    "reference_ref": {"plan_ref": "experiments/reference/acceptance_plan.json", "arm_id": "reference"},
    "purpose": "optimizer_comparison"
  }]
}
```

All names above are placeholders, not resources or permissions. `platform_ref`
contains the existing Spec `platform` object and Kaggle account/kernel/title/
datasets/source_dataset/accelerator object; it contains no credentials. Authority
must already cover exactly the arms, seed and positive full-allocation ceiling.
Reference recipe, data roles, initialization, target-transform asset, counters
and retained reference pins must agree. Missing/incompatible references block;
the compiler never retrains a baseline or creates authority. Ordinary catalog
addons use ordered `{name, version, config}` declarations; mixed/confounded
interventions fail the existing purpose-scoped comparison check.
For catalog addons with literal-only defaults, `config: null` selects those
reviewed defaults; bounded/configurable addons still require explicit values.
Addon names alone select version `1` and reviewed literal defaults. Inline
`authority`/`platform` may replace their `_ref` fields. Repeated arm fields may
live in `shared`; `preset_ref: {"path": "...", "sha256": "..."}` pins an existing baseline.
These shorthands preserve the same recipe and authority checks.

For inputs at `experiments/<experiment_id>/experiment.json`, omitted mechanical
fields resolve to `STATUS.md`, `protocol.md` and `decision.md` in that directory.
The compiler can derive the Kaggle block for the fixed 100K mount and a candidate
policy only when its ID matches the experiment ID with dashes. Review and commit
the generated policy. Scientific questions, authority and budgets remain explicit.
The pinned [K1 post-BN preset](../../experiments/presets/k1-post-bn-100k-v1.json)
reuses compact recipe/data/target contracts; supply the matching local initial state.
`supporting_evidence` contains RML evidence IDs; cite papers in the protocol.
A readiness result of `BLOCKED` before freeze leaves the logical run ID editable.
After preparation, `sandbox --spec SPEC --arm ARM --plan WORKFLOW --repo-root .`
uses the frozen initialization to inspect step-0 equality, gradients and serialization.

Commit human/source inputs before formal preparation. The tool generates only
bounded canonical JSON under `experiments/<question>/generated/<input+commit>/`;
these are compact definition metadata, not training payloads. Do not hand-edit
or commit generated envelopes. A V2 source-package manifest explicitly binds
their hashes and the committed human configuration; its six sidecars and V1
reader compatibility are preserved. The old baseline Spec is checked against
its Git blob; the new package's own Spec contains the complete selected data roles.
Run artifacts, source caches and scratch remain under the external machine roots.

The platform command in section 4 compiles and freezes this input. The supported CPU proof is
K1 and GPTrans parameter editing/preparation/recovery, not arbitrary recipe
compilation, new-family admission, native runtime qualification or training.
Changing an already frozen scientific input cannot overwrite its run/attempt.

Already authored declarations can use the reference-only V1 input in the
[CLI compatibility reference](EXPERIMENT_CLI.md#editable-inputs-and-frozen-records).
Other unsupported deltas retain their owning builders. `freeze-input` is an
optional freeze-only operation; it is not a separate required preparation step.

## 4. Prepare through the owning platform entry

Use the Kaggle workload skill. The same existing platform entry now accepts the
editable envelope and prepares without credentials or any API call:

```powershell
& $python platforms/local/run.py --profile $profilePath --script platforms/kaggle/submit_workflow.py -- --experiment $experiment --repo-root .
```

Outputs live at `<artifact_root>/runs/<logical_run_id>/<attempt_id>/`, with
`inputs`, `prepared` and `operations` separated. The fixed `runtime-v1` source
distribution automatically includes tracked model/addon Python; callers supply
scientific record references, not per-experiment Python lists. AST closure is
an optional development diagnostic, not a default packaging prerequisite.
Only source archives are cached; Specs, package identities, releases and receipts
are never borrowed from another run.
Packages containing generated run records reuse their prepared package directly
instead of retaining a second cache copy.

Preparation validates this question's references and freezes its knowledge
snapshot without rebuilding the global RML index. Index maintenance follows
the existing milestone/handoff boundary in the
[planning contract](EXPERIMENT_PROSPECTIVE_PLANNING.md).

With existing applicable submission authority, append `--credentials ...
--execute`. That calls the **original** journaled submitter. An uncertain outcome
requires reconciliation, never a fresh slug or blind retry. Completed preparation
is reused. Confirmed pre-publication local failures retain every failed byte and
regenerate staging with the same frozen identity; published/unknown RML effects
or any operation journal remain reconciliation-only. Source caches publish as
complete validated directories; retained legacy partial objects are rebuilt from
the same commit. `--prepared` and old
canonical `--spec` operations remain compatible. IMS/SCNet do not acquire this
new front door implicitly. See the [platform API](KAGGLE_WORKFLOW_SUBMISSION.md).

Both preparation entries resolve relative initialization paths against the
selected repository. One preparation, run lookup or acceptance transaction reuses full package validation
for identical bytes; each boundary still reads and hashes all six files.
Repository comparisons remain fresh; no validation memo survives the invocation.

Scientific-input continuation looks up the external frozen run **before**
compilation. The same human input reuses its original Spec/workflow/package,
even after unrelated HEAD changes or removal of development reference files,
generated definitions and original initialization paths. The completed package
and frozen bindings are still checked. New model source does not upgrade an old
run: choose an explicit new `logical_run_id` to create a new scientific task.
Changed input cannot overwrite a frozen run. Existing scientific freezes without
the newer source-commit pin remain readable; their original bindings are not migrated.

Unpublished preparation can regenerate only under its originally bound source
commit. A different HEAD or unavailable original source pin returns
`RECONCILIATION_REQUIRED` and preserves the failed directory. Use the original
source checkout to continue; this entry does not auto-switch Git, recreate
missing historical inputs, rewrite RML or clear journals. A completed prepared
package continues through the original journal, including uncertain submissions.

## 5. Inspect, accept and recover with existing owners

The run directory locates its frozen inputs, package and operation journal.
`locate-run` reads it without creating files or contacting a platform:

```powershell
& $python platforms/local/run.py --profile $profilePath -- locate-run --profile $profilePath --repo-root . --run-id $runId --attempt-id $attemptId
```

For registered runs, `platforms/kaggle/retrieve_family_outputs.py --all-arms
--execute` retrieves the exact receipt version's manifests, required artifacts
and execution-cost records; existing hash-matching files are reused. Its returned
`execution_root` is passed to acceptance. Without `--execute`, it plans locally.
Continuations use the original `--continuation-receipt` and `--resume-prepared`.
`--status` labels the latest scheduler observation separately from receipt identity.
A STOP_FOR_COST run is fetched with `--all-arms --paused` (unstrict, not an acceptance).

After retrieval, the local entry resolves frozen expectations and closure
locations, inspects each arm, writes attribution and stages terminal drafts:

```powershell
& $python platforms/local/accept_run.py --profile $profilePath --run-id $runId --attempt-id $attemptId --retrieved-root $retrievedRoot
```

`$retrievedRoot/<arm_id>` contains the already retained output files. Omit
`--retrieved-root` when the retriever has recorded those locations. Files stay
where they are; this command downloads and copies no training payloads. Drafts
do not close RML. Closure requires an explicit outcome, decision reference,
observed role evidence and `--execute` through the existing terminal owner.
Missing/uncertain receipt or journal bindings require original reconciliation.
For an already closed run, add `--strict --execute` to recheck its accepted
canonical evidence without rewriting the terminal. Closed continuations resolve
only their hash-bound accepted successor; unclosed continuations still require
an explicit receipt. The retriever reuses hash-matching outputs and compressed
native proofs; it never requests all platform files.
The [CLI reference](EXPERIMENT_CLI.md#local-lifecycle-by-run-id) owns the remaining
arguments and local path limits; the [recovery contract](REGISTERED_EXPERIMENT_WORKFLOW.md#recover-an-incomplete-workflow)
retains incomplete-checkpoint and native-runtime restrictions.

Scientific acceptance, complete allocation cost, role review, Replay, 500K/full
qualification and canonical RML closure are separate decisions. Missing facts
remain missing, not zero. No 100K score automatically rejects 500K; no old STQS
or uncalibrated early-stop rule acquires authority from this interface.

Preparation timing observations are retained outside frozen run bytes. Read the
named run's local samples and existing journal timings with:

```powershell
& $python platforms/local/run.py --profile $profilePath --module molgap.experiment_timings -- --run-directory $runDirectory
```

This read-only report does not qualify live latency. Active phase timers exclude
offline waiting; upload transfer, source-ready wait, end-to-end receipt latency
and training queue time remain unmeasured without their required observations.

## 6. Check affected flows and manage temporary storage

Select the affected workflow journeys in one pytest session; shared cases are deduplicated:

```powershell
& $python platforms/local/run.py --profile $profilePath --module molgap.local_testing -- --scope train --scope resume -- -q
```

Scopes are `runtime`, `package`, `prepare`, `train`, `submit`, `accept`, and `resume`.
`--list` previews fixed pytest nodeids, rather than every test in growing files.
Use ordinary pytest for an affected lower-level invariant. Follow the
[verification contract](INFRASTRUCTURE_VERIFICATION.md) for evidence limits and
test growth: strengthen one existing workflow instead of adding a file per
incident. Report the actual model, state transitions and substitutes exercised;
passing counts do not qualify the candidate's frozen package or native runtime.
Recheck affected failures after a fix; do not repeat unaffected successful suites.

The profile wrapper gives each invocation its own scratch and removes it on exit.
Save reports explicitly under `artifact_root`. To distribute tests after committing
source, use the existing source-bundle writer:

```powershell
& $python platforms/local/run.py --profile $profilePath --module molgap.local_testing -- --repo-root . --package E:/MolGap/artifacts/test-source
```

The absent/empty destination receives tracked source/tests and compact flow
fixtures, excluding datasets, weights and accounts. Transfer the hash/inventory
sidecars with the archive. From its extracted directory, use the host's dependency
interpreter and the same wrapper/scopes; no editable reinstall is needed.
Historical tests outside these scopes may require their own retained records.

Inspect bounded temporary-storage candidates without opening payloads:

```powershell
& $python platforms/local/run.py --profile $profilePath --module molgap.local_storage -- --profile $profilePath --repo-root .
```

Removal requires an exact `--candidate` and `--execute`. Only atomically retired
`.incomplete-*` source caches with known generated files are eligible; scratch of
unknown ownership, live caches and run evidence remain report-only. Interrupted
removal reports `STOPPED`. Age alone never establishes that a file is disposable.
`--report` summarizes artifact sizes and `--dedupe` reports identical-file groups;
hardlink mutation is disabled because resume writers can overwrite shared bytes.

Migration evidence, remaining capabilities and [server port acceptance](REFORM_MIGRATION.md#accepting-the-2026-11-desktop-reform-port-on-the-server-host)
route through the [cumulative reform record](REFORM_MIGRATION.md).

## Local execution and closure inputs

`platforms/local/train.py` runs one prepared local arm through the registered
trainer; `--pause-after-step` and reentry use its existing checkpoint. Its
`--accept-only` route supports `--closure-input FILE` and `--draft-role-audit DIR`,
and closes only `INFRASTRUCTURE_ONLY`. It grants no scientific or compute authority.

For run-keyed acceptance, `draft-closure` drafts the role audits, measured numbers,
closure facts and the `--closure-input` file for `accept_run.py --strict --execute`;
outcome and decision stay explicit judgments. Protected-role coverage and exact
arguments are in the [CLI reference](EXPERIMENT_CLI.md#local-lifecycle-by-run-id).
