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
  same addon there again. Build source-bound declarations with
  `build_addon_declaration(...)`.
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

## 3. Compile scientific inputs, or reuse an existing declaration

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

Commit human/source inputs before formal preparation. The tool generates only
bounded canonical JSON under `experiments/<question>/generated/<input+commit>/`;
these are compact definition metadata, not training payloads. Do not hand-edit
or commit generated envelopes. A V2 source-package manifest explicitly binds
their hashes and the committed human configuration; its six sidecars and V1
reader compatibility are preserved. The old baseline Spec is checked against
its Git blob; the new package's own Spec contains the complete selected data roles.
Run artifacts, source caches and scratch remain under the external machine roots.

Use the same single platform command in section 4. The supported CPU proof is
K1 and GPTrans parameter editing/preparation/recovery, not arbitrary recipe
compilation, new-family admission, native runtime qualification or training.
Changing an already frozen scientific input cannot overwrite its run/attempt.

For an already authored declaration, the reference-only V1 input remains valid:

The editable front door accepts normal JSON indentation and a final newline:

```json
{
  "format": "molgap-experiment-input-v1",
  "spec_ref": "experiments/question/experiment_spec.json",
  "workflow_ref": "experiments/question/workflow_plan.json",
  "machine_profile": "E:/MolGap/machine-profile.json",
  "authority_ref": "experiments/question/authority.json",
  "monitor_paths": ["experiments/question/STATUS.md"],
  "attempt_id": "v1"
}
```

These are illustrative references, not approved resources. The referenced Spec
may also be pretty JSON: freezing publishes the original canonical Spec identity.
The workflow must match that identity. Existing recipe, data, reference,
prospective and authority owners still validate applicability. No hypothesis,
authorization or acceptance is invented. A changed input cannot overwrite the
same frozen run/attempt.

V1 references existing declarations; the scientific format above compiles the
bounded supported deltas. Other new declarations retain their existing builders.
Frozen human/source records must be committed before formal packaging.

```powershell
& $python platforms/local/run.py --profile $profilePath -- freeze-input --experiment $experiment --repo-root .
```

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

Only after separate applicable authorization, append `--credentials ...
--execute`. That calls the **original** journaled submitter. An uncertain outcome
requires reconciliation, never a fresh slug or blind retry. Completed preparation
is reused. Confirmed pre-publication local failures retain every failed byte and
regenerate staging with the same frozen identity; published/unknown RML effects
or any operation journal remain reconciliation-only. Source caches publish as
complete validated directories; retained legacy partial objects are rebuilt from
the same commit. `--prepared` and old
canonical `--spec` operations remain compatible. IMS/SCNet do not acquire this
new front door implicitly. See the [platform API](KAGGLE_WORKFLOW_SUBMISSION.md).

## 5. Inspect, accept and recover with existing owners

The run directory locates its frozen inputs, package and operation journal.
`locate-run` reads it without creating files or contacting a platform:

```powershell
& $python platforms/local/run.py --profile $profilePath -- locate-run --profile $profilePath --repo-root . --run-id $runId --attempt-id $attemptId
```

Use the exact returned receipt and actual retrieved artifacts with
`accept-workflow`; use `build-resume` / `prepare-resume` only for a supported
incomplete run. These API arguments remain in the
[CLI reference](EXPERIMENT_CLI.md) and
[lifecycle contract](REGISTERED_EXPERIMENT_WORKFLOW.md).

Scientific acceptance, complete allocation cost, role review, Replay, 500K/full
qualification and canonical RML closure are separate decisions. Missing facts
remain missing, not zero. No 100K score automatically rejects 500K; no old STQS
or uncalibrated early-stop rule acquires authority from this interface.

For targeted tests, migration measurements, portable commits and remaining
work, see the [cumulative reform record](REFORM_MIGRATION.md). Do not run a full
historical/document-format audit for a routine model configuration change.
