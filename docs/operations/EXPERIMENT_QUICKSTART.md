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

## Change the scientific delta

- For shared K1/3 and GPTrans/2, the scientific input compiler accepts typed
  learning-rate/weight-decay changes and catalog addons. Existing builders still
  own other recipe changes, full-training contracts and platform admission.
- Add an ordinary addon once in `experiment_catalog.py`, with its typed fields,
  supported family/version and execution modes/hooks. `experiment_execution.py`
  derives its read-only adapters from that catalog; do not register it twice.
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
now accepts the editable input and prepares locally without an API call:

```powershell
& $python platforms/local/run.py --profile $profilePath --script platforms/kaggle/submit_workflow.py -- --experiment $experiment --repo-root .
```

Run files live at `<artifact_root>/runs/<logical_run_id>/<attempt_id>/`; `inputs`,
`prepared` and `operations` retain their distinct roles. `runtime-v1` packages
tracked runtime Python plus explicit scientific records. AST dependency discovery
remains an optional diagnostic. Source archives use a content cache; each run
still has its own Spec, package identity, release and receipts. No data or weight
download is implied by preparing a source package.

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

```powershell
& $python platforms/local/run.py --profile $profilePath -- locate-run --profile $profilePath --repo-root . --run-id $runId --attempt-id $attemptId
```

The family-output retriever can resolve this run plus independently pinned
manifest/arm identity. Without `--execute` it validates locally. Exact transport
remains in the existing platform adapter; historical output protocols keep their
own retrievers. `inspect-output`, `accept-workflow`, `build-resume` and
`prepare-resume` can resolve the original Spec/package/receipt from the run ID.
Supply the actual output/evidence inputs required by the existing owners.

The [CLI reference](EXPERIMENT_CLI.md#local-lifecycle-by-run-id) owns arguments;
the [workflow reference](EXPERIMENT_WORKFLOW.md) retains legacy preparation and
terminal semantics. Acceptance is read-only by default. Explicit terminal closure
still uses desktop's existing V5/RML pipeline. Mechanical acceptance, scientific
comparison, Replay, cost/roles, native qualification and full admission remain
separate. New configuration work needs focused tests, not a full historical audit
or document line-count checks.

The [desktop migration record](DESKTOP_INFRASTRUCTURE_UPGRADE.md) owns provenance
and executed verification. [Storage retirement](LOCAL_STORAGE_RETIREMENT.md)
owns payload and completed-worktree retirement.
