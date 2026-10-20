# Experiment infrastructure quickstart

This is the normal entry for new work. Read `AGENTS.md`, `CURRENT_STATE.md`, the
relevant `ROADMAP.md` section and the question's decision first. Query existing
RML before proposing a new experiment. This guide grants no compute, role,
scientific acceptance or cross-machine custody.

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

PPGT/3 and ESA/4 also use this front door through the
[configured Modern2D contract](MODERN2D_CONFIGURED_WORKFLOW.md), including
ordinary readout addons, explicit pass horizons and accepted indexed roles.
Their initial screens are context-only; the K1/GPTrans matched-reference example
below retains its own comparison checks.

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
selected repository. One preparation or run lookup reuses full package validation
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

For the family output protocol, the existing
`platforms/kaggle/retrieve_family_outputs.py` adapter accepts the same run ID,
profile, attempt and arm plus an independently pinned local/remote manifest and
an external artifact-root destination. Without `--execute` it validates locally
without authentication or creating the destination. Explicit credentials and
`--execute` call the original exact-file transport; unsupported historical
protocols still use their owning retriever. This is not scientific acceptance.

After the owning platform retrieves the exact attempt, `inspect-output`,
`accept-workflow`, `build-resume` and `prepare-resume` can resolve their frozen
Spec/package/receipt from the same run ID. Supply the actual outputs and
scientific/evidence inputs required by the existing owners:

```powershell
& $python platforms/local/run.py --profile $profilePath -- accept-workflow --profile $profilePath --repo-root . --run-id $runId --attempt-id $attemptId --outputs $outputs --locations $locations
```

Acceptance is read-only by default; explicit `--execute` invokes existing
terminal closure. Missing/uncertain receipt or journal bindings require original
platform reconciliation. No remote retrieval or scientific verdict is inferred.
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

Select the affected flows in one pytest session; overlapping files are deduplicated:

```powershell
& $python platforms/local/run.py --profile $profilePath --module molgap.local_testing -- --scope package --scope prepare -- -q
```

Scopes are `runtime`, `package`, `prepare`, `submit`, `accept`, and `resume`.
`--list` previews selections. Use ordinary pytest for other tests. These scopes
are conveniences, not additional release gates. Recheck affected failures after
a fix; do not repeat unchanged successful suites at every handoff.

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

Migration evidence and remaining capabilities route through the
[cumulative reform record](REFORM_MIGRATION.md).
