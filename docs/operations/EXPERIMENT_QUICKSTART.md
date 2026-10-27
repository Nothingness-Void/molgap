# Experiment infrastructure quickstart

Use this entry for new infrastructure work; read the owning question and query
RML before new science. Open the linked API references only for the operation
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

`molgap-scientific-experiment-v1` describes the question, existing authority,
reference recipe/data, initialization, supported delta, seed, platform and local
profile. The compiler generates Spec/workflow/prospective definitions and hashes
through existing owners. See the supported shape in
[`_compile_scientific_input`](../../src/molgap/experiment_input.py) and the
[synthetic fixture](../../tests/test_scientific_input.py); fixture identities
are examples, never real authorization or evidence.

`authority` and `platform` may be inline objects instead of `authority_ref` and
`platform_ref` (choose one form for each). Authority still names the exact arms,
seed and budget. Role and decision references retain real content. Addons accept
`"addon_name"` or `{"name":"addon_name"}` when the existing catalog supplies
version 1 defaults; required parameters still need explicit values.

An arm or `shared` may use `preset_ref: {"path":".../baseline-v1.json",
"sha256":"<frozen hash>"}` to reuse a reviewed versioned baseline. Its existing
compiler owner pins the recipe, data Spec/arm, target manifest and initialization
identity. `initial_state_ref` remains the explicit machine location. See the
[preset reader](../../src/molgap/experiment_presets.py) and
[executable examples](../../tests/test_scientific_input_ergonomics.py).
There is no mutable latest preset, implicit account/hardware template or automatic
weight download. Existing historical runtimes are not silently relabeled as new
K1/GPTrans presets.

For an explicitly authorized baseline/candidate job, `shared` supplies common arm
fields and each `arms` entry overrides whole fields. `same_run` binds
`reference_arm_id` and `target_manifest_ref`; authority must name that reference
arm in `same_run_reference_arm_id`. The reference keeps its frozen recipe and
the candidate supplies the scientific delta. This uses the existing same-run
acceptance owner; it does not require inventing a previously accepted reference.
See the executable [pair fixture](../../tests/test_scientific_pair_input.py).

A new `vN` attempt requires `retry_of` (logical run, predecessor attempt, Spec
identity and decision pointer), matching authority and an explicit authorized
terminal disposition. Submitted predecessors require an exact reconciled job
observation. Unknown publication cannot be retried through this input.

Already authored declarations can use this reference-only envelope:

```json
{
  "format": "molgap-experiment-input-v1",
  "spec_ref": "experiments/question/experiment_spec.json",
  "workflow_ref": "experiments/question/workflow_plan.json",
  "machine_profile": "D:/molgap-machine-profile.json",
  "authority_ref": "experiments/question/authority.json",
  "monitor_paths": ["experiments/question/STATUS.md"],
  "attempt_id": "v1"
}
```

The referenced authority must already cover the requested work. Missing reference,
dataset manifest or target identity blocks preparation; no baseline is retrained.
Commit human/source inputs before formal preparation. Generated definitions are
compact metadata, not weights or training data. Do not hand-edit them.

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
finalizer. The decision and role audit must describe observed facts; this route
does not establish scientific comparison or replay eligibility. Durable outputs
stay outside the checkout. Default output points to the report; `--verbose`
expands it.

```powershell
& $python platforms/local/run.py --profile $profilePath -- locate-run --profile $profilePath --repo-root . --run-id $runId --attempt-id $attemptId
```

The family-output retriever can resolve this run plus independently pinned
manifest/arm identity; `--all-arms` does that for every manifest already pinned
under `<run>/operations/remote_manifest/` in one call. Without `--execute` it validates locally. Exact transport
remains in the existing platform adapter; historical output protocols keep their
own retrievers. `inspect-output`, `accept-workflow`, `build-resume` and
`prepare-resume` can resolve the original Spec/package/receipt from the run ID.
Supply the actual output/evidence inputs required by the existing owners.
`prepare-resume` returns `RESUME_INPUTS_PREPARED` for local continuation inputs.
The owning continuation adapter must bind a physical successor; this result is
not a registered new attempt or a fresh-release Kaggle submission.

For a retrieved frozen run, the local acceptance script resolves every arm's
expectations and prospective locations, inspects the artifacts, diagnoses the
available pairs and stages terminal drafts:

```powershell
& $python platforms/local/accept_run.py --profile $profilePath --run-id $runId --attempt-id $attemptId --retrieved-root D:/molgap-artifacts/retrieved/$runId
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
RunContext and existing role events, not merely the recipe's permissions. Closure
uses the existing V5/RML validators and reports missing evidence; positive/READY
claims still require their owning scientific acceptance. The CLI reference owns
the advanced explicit-location path.

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
whole repository's indexes or requiring unrelated historical weights. Use the
existing RML `rebuild` at handoff/accepted milestones. Multi-arm closure
persists each independent terminal outcome and updates derived indexes once.
If that update fails, use the RML CLI `terminal-rebuild --trajectory <path>` to
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
`--report` shows where `artifact_root` bytes sit and `--dedupe` hardlinks
byte-identical copies without removing any path.
