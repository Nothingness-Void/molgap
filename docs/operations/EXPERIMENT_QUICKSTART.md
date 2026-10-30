# Experiment infrastructure quickstart

For routine operations on an already declared, authorized run, start with its
platform skill and supplied contract/attempt records; use only the relevant
commands and references. Use this entry for new infrastructure work; read the
owning question and query RML before new science. Open the linked API references only for the operation
being changed. A configuration or successful preparation does not grant compute,
protected roles, scientific acceptance or production promotion.

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

```powershell
$python = 'D:/文档/molgap/.venv/Scripts/python.exe'
$profilePath = 'D:/molgap-machine-profile.json'
& $python platforms/local/run.py --profile $profilePath -- catalog
```

The local wrapper selects this checkout in an isolated child and reports its
actual interpreter and MolGap module. Do not redirect the shared editable install.
With a machine profile, each child owns a temporary subdirectory of `scratch_root`;
normal success, nonzero exit and launch exceptions clean that subdirectory. Keep
durable outputs and diagnostic logs in `artifact_root`, outside temporary scratch.
Forced process termination can still leave scratch behind.

## Change the scientific delta

- For shared K1/3 and GPTrans/2, the scientific input compiler accepts typed
  learning-rate/weight-decay changes and catalog addons. Existing builders still
  own other recipe changes, full-training contracts and platform admission.
- Add an ordinary addon once in `experiment_catalog.py`, with its typed fields,
  supported family/version and execution modes/hooks. `experiment_execution.py`
  derives its read-only adapters from that catalog; do not register it twice.
  Shared recipe versions, modes and output profiles follow those adapters.
  Multiple versions for the same shared recipe owner require an explicit routing
  decision; the projection does not silently choose the newest version.
- Keep input ABI, objective and parameter ownership explicit. Reuse the family
  trainer and [runtime extension API](MODULAR_EXPERIMENT_RUNTIME.md). Use the
  [operation map](EXPERIMENT_ADDON_GUIDE.md#pick-the-operation) to find the owner.

Desktop's existing family/addon declarations, accelerated defaults, full/SCNet
paths and native training implementations are preserved. Catalog registration
does not establish native qualification. The two deferred model modules are not
part of this migration.

## Supply one human input

The infrastructure holds the framework. For a K1 post-BN 100K pair you fill in
only the experiment's own details: question, arms, the catalog addon and its
parameters, authority and budget. Put the file at
`experiments/<experiment_id>/experiment.json`, commit it with a non-empty
`protocol.md` and `decision.md` beside it, and run `submit_workflow.py` (below).

```json
{
  "format": "molgap-scientific-experiment-v1",
  "experiment_id": "pcqm_k1_my_question_100k",
  "logical_run_id": "pcqm-k1-my-question-100k-s42",
  "attempt_id": "v1",
  "machine_profile": "D:/molgap-machine-profile.json",
  "timestamp": "2026-10-29T00:00:00Z",
  "question": { "policy_id": "pcqm-k1-my-question-100k", "action": "RUN_PAIRED_100K", "question": "...",
    "deficiency": "...", "changed_mechanism": "...", "supporting_evidence": ["..."],
    "alternatives": ["..."], "closed_families": ["..."],
    "positive_decision": "...", "negative_decision": "..." },
  "profile": { "owner": "desktop", "track": "B", "platform": "kaggle",
    "hardware": "Tesla T4x2 FP32, one arm per device",
    "cheapest_falsifier": "...", "estimated_device_hours": 3.7, "estimated_wall_hours": 1.85 },
  "authority": { "format": "molgap-execution-authority-v1", "user_authorized": true,
    "account": "<kaggle account>", "arms": ["baseline", "candidate"], "seeds": [42],
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
initialization digest and model family. The compiler derives the rest from the
experiment directory: `monitor_paths` (`STATUS.md`), `role_ref` (`protocol.md`),
`decision_ref` (`decision.md`), the Kaggle block (kernel, title and source dataset
from the authority `account` and the experiment ID, plus the fixed read-only
`<account>/pcqm4mv2-ogb-fixed-100k-v1` mount) and, when `policy_id` is not
registered and equals the experiment ID with dashes, a candidate `research_memory/policies/<policy_id>.1.json` whose
source digest is the `protocol.md` digest. Review and commit that policy file.
Any of these fields written explicitly in the input wins. The question, arms,
addon, authority, scope and budget have no default and are never invented.
`supporting_evidence` takes RML evidence ids only; cite papers in `protocol.md`.
A prepare that fails after the run is frozen needs a new `logical_run_id`
(`...-r2`) for any corrected input. `sandbox` reads the frozen inputs, so run it
after prepare and before submitting.
Check an addon on the real frozen state before submitting:

```powershell
& $python platforms/local/run.py --profile $profilePath -- sandbox --spec $run/inputs/experiment_spec.json --arm candidate --plan $run/inputs/workflow_plan.json --repo-root .
```

Known limits: Kaggle T4x2 only; seed 42 only; one reference plus one candidate;
typed edits limited to `learning_rate` and `weight_decay`; the derived Kaggle
block supports only the fixed 100K mount (write `platform` explicitly otherwise);
a new `vN` attempt needs `retry_of` with matching authority and a reconciled
terminal disposition. Fixture identities in
[`tests/test_scientific_input_ergonomics.py`](../../tests/test_scientific_input_ergonomics.py)
are examples, never authorization or evidence.

## Prepare through the owning platform

Use the Kaggle workload skill for its platform operations. This existing adapter
accepts the editable input. With existing submission authority, one invocation
prepares and advances the original journal:

```powershell
& $python platforms/local/run.py --profile $profilePath --script platforms/kaggle/submit_workflow.py -- --experiment $experiment --repo-root . --credentials $credentials --execute
```

Omit `--execute` and `--credentials` for local preparation when a separate review
is useful. A dry pass is optional; preparation already performs its admission
checks. Do not chain every diagnostic command before the same submission.

Run files live at `<artifact_root>/runs/<logical_run_id>/<attempt_id>/`; `inputs`,
`prepared` and `operations` retain their distinct roles. `runtime-v1` packages
tracked runtime Python plus explicit scientific records. AST dependency discovery
remains an optional diagnostic. Source archives use a content cache; each run
still has its own Spec, package identity, release and receipts. No data or weight
download is implied by preparing a source package.

Both preparation entries resolve relative initialization paths against the selected
`repo_root`, independent of the caller's working directory. Within one invocation
from preparation through submission, identical package bytes reuse full validation;
each boundary still reads and hashes all six package files. Independent calls
start fresh. A byte-identical upload roundtrip may reuse expensive content
checks inside that synchronous submission; remote identity, layout, entry and
mount checks still run. No verification result is persisted between preparations.

Preparation commands print compact status, identity, next action and report/journal
pointers by default; `--verbose` exposes the existing complete report. Re-reading
an already prepared run does not create another timing observation file.

With existing submission authority, the platform's `--execute` path uses its
original journaled submitter. Unknown publication outcomes require reconciliation.
Old `--spec` and `--prepared` APIs remain available. IMS/SCNet retain their existing
adapters and contracts; they do not implicitly acquire this front door.

Continuation resolves the original frozen run before compiling new input. The
same input reuses its Spec/package/journal across unrelated HEAD changes. Changed
input needs a new logical run. Incomplete unpublished preparation can regenerate
only from its bound source commit; uncertain effects or journals remain
`RECONCILIATION_REQUIRED`. Never clear a journal to force a retry.

## Retrieve, accept and recover

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
finalizer. `--closure-input FILE` (same shape as the run-keyed one, a single arm,
outcome `INFRASTRUCTURE_ONLY`) replaces the three terminal flags, and
`--draft-role-audit <repo-dir>` drafts the audit from the retained artifacts
(`--protected-role-use ROLE=STATE` only from observed facts; unknown blocks closure).
The drafted audit is published immutably, so a rerun with different role states
needs a new `--draft-role-audit` directory.
The run-keyed `accept_run.py`, `draft-role-audit` and `diagnose` need an
`inputs/input_binding.json` that a local run does not have, so use these flags here. The decision and role audit must describe observed facts; this route
does not establish scientific comparison or replay eligibility. Durable outputs
stay outside the checkout. Default output points to the report; `--verbose`
expands it.

```powershell
& $python platforms/local/run.py --profile $profilePath -- locate-run --profile $profilePath --repo-root . --run-id $runId --attempt-id $attemptId
```

The Kaggle retriever's `--all-arms --execute` resolves the exact receipt version,
fetches missing small manifests, then retrieves every arm and execution-cost
records through the existing transport. It returns `execution_root` for acceptance
and records per-arm destinations. `--status` observes the latest scheduler state,
labelled separately from the receipt version, without downloads or submission.
No manual manifest/SHA file is needed for the
registered `experiment/<arm>` layout. Strict continuation also supplies
`--continuation-receipt` and `--resume-prepared`; historical layouts may set
`--remote-root`. Without `--execute`, all-arm retrieval only plans locally.
Per-arm retrieval remains available for explicit historical pins.
`inspect-output`, `accept-workflow`, `build-resume` and `prepare-resume` resolve
the original Spec/package/receipt from the run ID.
Supply the actual output/evidence inputs required by the existing owners.
`prepare-resume` returns `RESUME_INPUTS_PREPARED` for local continuation inputs.
The owning continuation adapter must bind a physical successor; this result is
not a registered new attempt or a fresh-release Kaggle submission.

For a retrieved frozen run, the local acceptance script resolves every arm's
expectations and prospective locations, inspects the artifacts, diagnoses the
available pairs and stages terminal drafts:

```powershell
& $python platforms/local/accept_run.py --profile $profilePath --run-id $runId --attempt-id $attemptId
```

Omit `--retrieved-root` after the retriever has recorded verified per-arm
destinations. Explicit `--outputs` remains available; its paths cannot override
frozen expectations. Default diagnostic reports are shared once under
`<run_dir>/reports/diagnostic_<content-identity>.json`; identical retries reuse
the same path. Stdout is compact (`--verbose` for
details). Neither this script nor diagnosis downloads data or trains a baseline.

For already closed continuation runs, `--strict` resolves the common successor
receipt from all arms' hash-bound canonical continuation records and selects
that version's retained outputs. It never guesses the newest version. Explicit
`--continuation-receipt` remains available for an unclosed continuation. Repeating
strict acceptance of eligible canonical results is read-only even with `--execute`;
it does not regenerate terminal drafts or close the same run again.

Drafts preserve unknown role/cost/scientific facts. Add an explicit `--outcome`
and repository-local `--decision-ref` to stage a decision; `--execute` alone
cannot infer one. For several arms, `--closure-input FILE` carries each arm's
outcome and decision path (and an optional role audit) in one human-authored file
instead of repeated flags or hand-written code; see the CLI reference. Where the producer lacks observed role truth, `--role-audit`
supplies a retained audit or an arm-to-audit-path JSON map. Audits bind the exact
RunContext and existing role events, not merely the recipe's permissions.
`draft-role-audit` drafts those files from retained artifacts; protected-role
states are never derived and stay `unknown` (blocking closure) unless you pass them. Closure
uses the existing V5/RML validators and reports missing evidence; positive/READY
claims still require their owning scientific acceptance. The CLI reference owns
the advanced explicit-location path.

For first closure after continuation, pass the same `--continuation-receipt`
to retrieval, `draft-role-audit` and acceptance. Acceptance also takes the existing
`--resume-prepared` and applicable recovery authority; the original run identity stays unchanged.

When the execution-retention bundle was retrieved, pass `--execution-root` to
bind its verified allocation ledger into terminal evidence and cost events.
Only each arm's assigned, observed window is counted; prior segments and unknown
intervals remain explicit. An existing measured native allocation is not counted
again. The ledger is validated once per acceptance call; finalization still
checks the bound bytes independently.

The [CLI reference](EXPERIMENT_CLI.md#local-lifecycle-by-run-id) owns arguments;
the [workflow reference](EXPERIMENT_WORKFLOW.md) retains legacy preparation and
terminal semantics. Acceptance is read-only by default. Explicit terminal closure
still uses desktop's existing V5/RML pipeline. Mechanical acceptance, scientific
comparison, Replay, cost/roles, native qualification and full admission remain
separate. New configuration work needs focused tests, not a full historical audit
or document line-count checks.

New RML plans share an immutable decision-time knowledge/policy snapshot. Arm
records bind its digest; legacy inline records remain readable. Preparation validates
the question's references and publishes canonical plans without rebuilding the
whole repository's indexes or requiring unrelated historical weights. Multi-arm
closure persists each independent terminal outcome and updates derived indexes
once; do not run another rebuild after successful closure. Standalone canonical
edits or imports use one RML `rebuild` after the accepted milestone.
Only if closure's derived update actually fails, use the RML CLI
`terminal-rebuild --trajectory <path>` to
recover from canonical receipts without repeating training or submission.

Canonical pass curves record the actual last-update learning rate separately
from the scheduler's next rate, and bind the published checkpoint. Historical
unknown fields remain unknown. These fields alone do not establish strict replay.

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
Combine affected scopes in one invocation. Direct pytest remains available for
affected lower-level invariants. Selection filters/skips mean partial coverage;
these journeys are not a mandatory release gate or native qualification.

Run relevant tests after the final code changes. Re-run a failing or affected
selection after a fix; do not repeat successful unchanged selections at every
handoff, commit or merge. Broaden coverage when the change crosses another owner.
The profile wrapper removes its own test scratch on exit; save any needed pytest
logs/JUnit reports explicitly under `artifact_root` before returning. Do not create
another worktree or environment merely for a test stage.

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
datasets, accounts and dependency environments. Other historical tests may need
their own repository evidence; this is not a full experiment archive.

Transfer the bundle and its existing hash/inventory sidecars, then extract
`source.tar.gz` into an empty directory. On server, use its dependency environment
with pytest and Git installed and run from that directory. Tests create their own
temporary Git fixtures; the extracted directory does not need Git metadata:

```bash
python platforms/local/run.py --module molgap.local_testing -- --scope runtime -- -q
```

Select the affected scopes as usual. The wrapper selects the extracted source;
no editable reinstall or new worktree is required. Test distribution is separate
from the training `runtime-v1` inventory, which retains its worker boundary.

The [desktop migration record](DESKTOP_INFRASTRUCTURE_UPGRADE.md) owns provenance
and executed verification. [Storage retirement](LOCAL_STORAGE_RETIREMENT.md)
owns payload and completed-worktree retirement. After a run closes, its
`--report` shows where `artifact_root` bytes sit. `--dedupe` is also read-only;
hardlink mutation is disabled because resume writers may overwrite shared bytes.
