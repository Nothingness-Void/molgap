# Modular Experiment Workflow

The [quickstart](EXPERIMENT_QUICKSTART.md) owns the default workflow. This page
is the preparation, acceptance, recovery and allocation-retention API reference.
Family trainers own execution; platform adapters own submission/retrieval;
scientific and native qualifications retain their existing owners.

## Owners

| Layer | Owner | Responsibility |
|---|---|---|
| Lifecycle | `experiment_workflow.py`; `prepare-workflow` / `accept-workflow` in the local CLI | Reuse package, release-check, prospective-planning, family-output inspection and existing terminal closure owners. |
| Family execution | `experiment_catalog.py` -> derived `experiment_execution.py`; `experiment_training_worker.py` | Single-source static dispatch and isolated subprocesses. No caller-selected import or callback. |
| Launch boundary | `experiment_launch_config.py` | Complete schema, immutable package/archive, per-arm recipe/init/prospective and unique device assignments; reused locally and by the frozen bootstrap. |
| Inspection | `experiment_inspection.py`; `experiment_family_workflow.py` | One CPU inspection produces an immutable snapshot; descriptor/closure recheck file hashes without deserializing tensors again. |
| Recovery | `experiment_resume.py`; `experiment_workflow_resume.py` | Hash-bound incomplete checkpoint transport and local staging against the original frozen release; family owners validate optimizer/RNG/cursor semantics. |
| Allocation retention | `experiment_allocation.py`; `experiment_retention.py` | Account for the full observed physical allocation and seal its compact execution records separately from scientific artifacts. |
| Platform preparation | Static platform registry in `experiment_workflow.py`; `kaggle_workflow.py` | Platform-owned plan validation, input staging/freezing and final release binding. Kaggle is the only registered preparation adapter in this entry point. |
| Kaggle runtime | `platforms/kaggle/run_experiment.py`; `kaggle_pair_runtime.py` | Verify the frozen launch/package and observed T4 allocation; preflight every arm before spawning any training arm. |
| Platform operations | `kaggle-molgap-workloads` skill and existing Kaggle adapters | Publish the prepared dataset/kernel, submit, reconcile the exact run/version, and retrieve retained outputs. |

`prepare-workflow` includes readiness checks. `check-workflow` is available when
only a read-only result is needed, without staging or publication.
Prepared bounded T4 releases use the journaled
[platform submission entry](KAGGLE_WORKFLOW_SUBMISSION.md).

The all-arm barrier runs the registered preflight phase for every assigned arm.
Training starts only after every preflight exits successfully. A failure stops
the phase and retains per-arm logs and pair state. This is orchestration
behavior; it is not evidence of real-shard or GPU qualification.

## Supported execution registry

`ExperimentSpec` declaration support does not imply executable training
support. This workflow requires Spec v2 and fails closed unless each arm matches
the static training registry, its mode, and its output profile.

Query `catalog` for the single `experiment_catalog.py` projection and
`workflow-info --spec SPEC` for the selected execution modes, output profiles
and preparation entry. Executable adapters are derived in
`experiment_execution.py`; output profiles live in `experiment_family_artifacts.py`.
`DECLARATION_ONLY` and `CPU_SYNTHETIC_ONLY` registrations do not release native
training. Unsupported families, addons, modes and allocations stop before execution.

## Add an addon

The [addon ownership API](EXPERIMENT_ADDON_GUIDE.md#addon-ownership) defines
the single catalog entry, declaration/recipe helpers and focused checks.
Normal scientific inputs invoke those helpers; manual recipe/Spec construction
is needed only for an owning API not covered by the compiler. Legacy family
versions/output profiles keep their original semantics. Reuse an accepted
reference unless the scientific contract explicitly requires retraining.

## Prepare once

For the default `--experiment` route, preparation generates or resolves these
records; see [editable inputs](EXPERIMENT_CLI.md#editable-inputs-and-frozen-records).
Direct `prepare-workflow` callers supply ordinary UTF-8 JSON with top-level
`format`, `spec_identity`, `source_profile`, `source_records`, `arms`,
`acceptance_plan`, `kaggle`. `source_profile: "runtime-v1"` includes tracked
runtime Python automatically. `source_records` names actual frozen scientific
records. Scientific compilation additionally binds `generated_records` and
`configuration_source`; callers do not hand-edit those machine records.

Legacy plans replace `source_profile`/`source_records` with `source_files`,
repository-relative POSIX extras to the registered inventory. Arm recipes are
included automatically. Keep that schema for frozen legacy packages; new inputs
do not require manual Python lists or AST discovery.

Every Spec arm appears once. `recipe` is a pinned repository-relative POSIX path;
`initial_state` is a pinned existing local file; `device` is distinct and within
the declared range. `acceptance_plan` names the full pinned family plan under
the repository root. Field contracts remain in [Spec](EXPERIMENT_SPEC.md),
[prospective planning](EXPERIMENT_PROSPECTIVE_PLANNING.md) and
[family acceptance](EXPERIMENT_FAMILY_WORKFLOW.md).

The `kaggle` object has exactly `account`, `kernel`, `title`, `datasets`,
`source_dataset`, and `accelerator`. The kernel must equal the account plus the
title-derived slug; title length is 5–50 characters. Dataset slugs must be
unique and include the account-owned source dataset. This workflow supports
the explicit `NvidiaTeslaT4` allocation only.

```json
{
  "format": "molgap-experiment-workflow-v1",
  "spec_identity": "<identity returned by validate-spec>",
  "source_profile": "runtime-v1",
  "source_records": ["experiments/<question>/authority.json"],
  "arms": [{
    "arm_id": "<Spec arm id>",
    "device": 0,
    "recipe": "experiments/<question>/training_contract.json",
    "initial_state": "E:/MolGap/artifacts/frozen/<arm>-initial-state.pt"
  }],
  "acceptance_plan": "experiments/<question>/family_acceptance_plan.json",
  "kaggle": {
    "account": "<account>",
    "kernel": "<account>/<title-derived-slug>",
    "title": "<5-50 characters>",
    "datasets": ["<account>/<source-dataset>", "<owner>/<fixed-dataset>"],
    "source_dataset": "<account>/<source-dataset>",
    "accelerator": "NvidiaTeslaT4"
  }
}
```

`prepare-workflow` requires a fresh output directory. It checks every arm and
the pinned acceptance inputs, builds the immutable source package, runs the
family's static recipe checks, and runs `check_release_inputs`. These checks
bind the fixed recipe, initialization, sampler, roles, exposure and packaged
bootstrap before prospective publication. The prospective planner validates
all arms before publishing. The workflow then freezes the launch config and
runs `check_workflow_binding` over the config, staged dataset mounts, T4 and
device shape, kernel metadata, prospective trajectory hashes and final entry
script hash. `release_report.json` records this final binding; the Kaggle
adapter rejects a missing, failed or stale report and repeats the byte checks
before POST. The output contains `package/`, `source_dataset/`, `kernel/`,
`release_report.json`, and `workflow_report.json`.

`workflow_report.json.timings` records local validation, packaging, release
checking, prospective publication and final binding durations. Static family
recipe and all-arm prospective checks precede packaging and publication.

The [CLI reference](EXPERIMENT_CLI.md#registered-family-lifecycle) gives direct
command arguments. Prepared payloads retain the platform's final release report;
the owning adapter rechecks it before POST and retains the observed account,
run reference and version. A local receipt or queue response is not acceptance.
Existing `--prepared` submission remains supported by the
[platform entry](KAGGLE_WORKFLOW_SUBMISSION.md).

If preparation returns `RECONCILIATION_REQUIRED` or prospective publication
fails partway, preserve the report and reconcile any published trajectories
before retrying. A changed executable source commit needs a newly reconciled
plan. No automatic retry or remote action is performed.

## Accept retained outputs

After the platform skill reconciles the exact attempt and retrieves the
manifest-bound outputs, provide:

- `outputs.json`: every Spec arm maps to exactly `{ "output_dir": ..., "expected": ... }`.
  `expected` is the same frozen seven-field contract owned by the family
  workflow: exposure counts, development row count, row/target hashes and
  precision.
- `locations.json`: every arm maps to existing `trajectory_id`, `run_id`,
  `trajectory`, `terminal`, and `trace` locations. These point to canonical
  per-arm files under the repository. The trace location must equal the
  inspected output trace path and bytes.
- The reconciled `receipt.json`, exact source `package/`, and its package
  identity. Acceptance requires observed physical-arm mapping and platform
  version; it will not guess either.

```powershell
& $python -m molgap.experiment_cli accept-workflow --spec $spec --repo-root $repoRoot --package $package --expected-package-identity $packageIdentity --receipt $receipt --outputs $outputs --locations $locations
```

Without `--execute`, all arms are inspected and the result is
`MECHANICALLY_VERIFIED`; the command returns the translated descriptor but does
not write terminal/RML records. Add `--execute` only when the owning terminal
inputs are ready to run the existing closure pipeline. `COMPLETE` means that
pipeline completed; scientific acceptance, comparison, runtime qualification,
role/cost review and promotion retain their existing owners and gates. A
blocked arm blocks closure for the whole workflow.

For failed, cancelled or interrupted work, use the CLI's incomplete-terminal
path with actual scheduler observations and existing canonical per-arm
locations. `kaggle_pair_runtime.py` retains `pair_state.json`; the CLI can
translate that state when every arm is incomplete. Formal progress is zero only
when the runtime recorded that training never started; counters after training
starts remain unknown if they were not retained. Mixed complete/incomplete
pairs use the existing per-arm inspection and terminal route. See
[terminal evidence](EXPERIMENT_CLI.md) for command fields and limits.

## Recover an incomplete workflow

Reconcile the exact original platform run/version and retrieve each arm's
checkpoint, trace, available selected state/predictions, runtime/provenance
sidecars and allocation ledger using the owning workload skill. Recovery uses
the original package and exact ACTIVE canonical prospective bytes. It never
creates another prospective action or changes the scientific contract.

```powershell
& $python -m molgap.experiment_cli build-resume --spec $spec --repo-root $repoRoot --prepared $originalPrepared --package $package --expected-package-identity $packageIdentity --receipt $originalReceipt --arm $armId --source-output $retainedArm --output $freshArmBundle
& $python -m molgap.experiment_cli prepare-resume --spec $spec --repo-root $repoRoot --prepared $originalPrepared --package $package --expected-package-identity $packageIdentity --receipt $originalReceipt --resume-plan $resumePlan --output $freshRecovery
```

The resume plan has exactly `format: "molgap-workflow-resume-v1"`,
`spec_identity`, and `arms: {"<arm-id>": "<bundle-directory-or-manifest-path>"}`.
Every Spec arm must appear. The transport validates hashes and identity; the
registered family validates model/optimizer/scheduler/RNG/cursor and native
runtime consistency. Preparation stages those exact bytes and freezes each
manifest digest into the launch. The final release gate and frozen bootstrap
repeat resume validation before any arm starts. Runtime restores fresh arm
outputs before the all-arm preflight barrier and retains prior cost segments.
Recovery uses a captured prospective byte snapshot and rejects canonical
changes during preparation. Mutable JSON sidecars, including allocation
ledgers, bind the hash and retained copy to one captured byte snapshot; later
atomic accounting refreshes do not change that bundle. Restoration verifies
the copied bytes against the manifest before atomic publication and rejects
concurrent destination conflicts.

Supported recovery requires all arms to be incomplete and checkpoint-bearing.
Completed arms, missing checkpoints, mixed completed/incomplete sets, changed
prospective records and legacy packages without the recovery runtime are
rejected. Use the existing owning per-arm route for those cases. Never modify
an old archive to add recovery support.
The local family validator must match its frozen packaged source, and the new
worker runtime must match the retained certificate. Releases with additional
pickled input staging remain with their owning preparation adapter.

## Retain allocation observations

Runtime writes `allocation_ledger.json` at the execution root and each arm,
then seals `execution_retention.json` and `execution_report.json` separately
from family scientific manifests. The ledger measures the Python bootstrap and
runtime observation window for every physical device, including unassigned
and idle devices. Queue and provisioning before Python remain explicitly
unmeasured. A prior running ledger is preserved as an incomplete observation;
recovery does not invent the unobserved interval after its last write.
The running ledger is written atomically on a 30-second interval while the
parent orchestrator is alive; terminal retention is sealed after completion or
a handled failure.
Prior invocations are flattened and deduplicated, preserving incomplete
observation flags across repeated recovery.

Retrieve the independently pinned execution retention manifest and only its
bound compact files through the Kaggle adapter. Supply `--execution-root ROOT`
to `accept-workflow` to validate their source/package/Spec identity and hashes
alongside scientific output inspection. Execution retention is a cost
observation, not a replacement for the existing scientific/native-cost gate.

## Evidence boundary

Package/release checks and synthetic local fixtures establish interface and
mechanical behavior only. They do not establish real-shard execution, GPU
qualification, scientific validity or training gains. The platform skill owns
remote state and retrieval; the family contract and RML/V5 owners govern
acceptance. See [architecture](../../ARCHITECTURE.md) and
[family output checks](EXPERIMENT_FAMILY_WORKFLOW.md).


## Server compatibility

This integration retains server preparation, EdgeState output profiles, author
modes, exact prospective attempt IDs and full independently frozen metric
semantics. Recipe builders add `metric_semantics`; an EMA-only development
trace is allowed when that exact null/live/EMA definition is frozen beforehand.
`build-terminal` requires `--repo-root` and reads the owning prospective action
for its attempt ID. A platform version is never used to invent an attempt ID.
The existing [server workflow](EXPERIMENT_WORKFLOW.md) owns server release and
strict scientific/replay qualification; generic local preparation is not that gate.
