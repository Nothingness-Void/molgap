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

- Supported parameters: use the owning recipe/configuration builder; preserve
  its exact batch, sampler, objective, normalization and budget semantics.
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

## 3. Reference and freeze existing scientific records

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

This first front door references existing declarations. It does **not** compile
an arbitrary short model/recipe override into a new scientific experiment;
new declarations still use the existing builders and prospective schema.
Frozen recipe/authority/plan source records must be committed before packaging.

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
is reused; incomplete preparation is retained and blocked. `--prepared` and old
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
