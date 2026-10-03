# Modular Experiment Workflow

Use the [infrastructure quickstart](EXPERIMENT_QUICKSTART.md) for the shortest
new-agent route. This document owns the detailed registry, preparation,
acceptance, recovery, and allocation-retention contract; it does not authorize
remote submission.

This is the entry point for the registered local-to-Kaggle training path. It
owns preparation and acceptance orchestration. It does not own scientific
authorization, a trainer loop, or platform submission.

For a supported Spec, query `workflow-info` before selecting the preparation
entry. Registration and local preparation still require the owning scientific
contract and the platform's actual runtime qualification.

## Owners

| Layer | Owner | Responsibility |
|---|---|---|
| Lifecycle | `experiment_workflow.py`; `prepare-workflow` / `accept-workflow` in the local CLI | Reuse package, release-check, prospective-planning, family-output inspection and existing terminal closure owners. |
| Family execution | `experiment_execution.py`; `experiment_training_worker.py` | Static family/addon-to-trainer dispatch and one isolated subprocess per arm/phase. No caller-selected import or callback. |
| Launch boundary | `experiment_launch_config.py` | Complete schema, immutable package/archive, per-arm recipe/init/prospective and unique device assignments; reused locally and by the frozen bootstrap. |
| Inspection | `experiment_inspection.py`; `experiment_family_workflow.py` | One CPU inspection produces an immutable snapshot; descriptor/closure recheck file hashes without deserializing tensors again. |
| Recovery | `experiment_resume.py`; `experiment_workflow_resume.py` | Hash-bound incomplete checkpoint transport and local staging against the original frozen release; family owners validate optimizer/RNG/cursor semantics. |
| Allocation retention | `experiment_allocation.py`; `experiment_retention.py` | Account for the full observed physical allocation and seal its compact execution records separately from scientific artifacts. |
| Platform preparation | Static platform registry in `experiment_workflow.py`; `kaggle_workflow.py` | Platform-owned plan validation, input staging/freezing and final release binding. Kaggle is the only registered preparation adapter in this entry point. |
| Kaggle runtime | `platforms/kaggle/run_experiment.py`; `kaggle_pair_runtime.py` | Verify the frozen launch/package and observed T4 allocation; preflight every arm before spawning any training arm. |
| Platform operations | `kaggle-molgap-workloads` skill and existing Kaggle adapters | Publish the prepared dataset/kernel, submit, reconcile the exact run/version, and retrieve retained outputs. |

The all-arm barrier runs the registered preflight phase for every assigned arm.
Training starts only after every preflight exits successfully. A failure stops
the phase and retains per-arm logs and pair state. This is orchestration
behavior; it is not evidence of real-shard or GPU qualification.

## Supported execution registry

`ExperimentSpec` declaration support does not imply executable training
support. This workflow requires Spec v2 and fails closed unless each arm matches
the static training registry, its mode, and its output profile.

| Family/version | Reference mode | Registered addon mode(s) | Output profile |
|---|---|---|---|
| `neural_atom_k1/2` | `reference` | `k1_joint_aggregation/1` → `ssma` | `k1-screen-v1` |
| `gptrans_t/1` | `reference` | `pair_prenorm/1`, `centered_logits/1`, `memory_value/1`, `memory_message/1` | `gptrans-v1` |

The executable registry is in `experiment_execution.py`; the output profiles
are in `experiment_family_artifacts.py`. `experiment_source_inventory.py`
owns reviewed common bootstrap files. Each family adapter declares its source
and serialized-module dependencies; an addon declares only additional reviewed
dependencies. Packaging selects the registered families/addons in the Spec.
The family trainer or shared graph owner supplies model/input semantics,
optimizer, selection and resume behavior. A same-family variant supplies only
its registered addon set, frozen contract/recipe and config; legacy adapters
retain their declared single-addon limits, while the generic graph adapter may
apply multiple approved hooks in registry order. It does not add an
experiment-specific training loop or launcher. A new graph family registers its
declarative family contract, `graph_training_adapter` model factory, output
profile and reviewed source inventory once. A new addon adds its reviewed mode
or `apply_addon` hook and source path only when it introduces a new dependency.
Unsupported families, addon versions, modes and platform allocations stop before
execution. Declaration support alone is not runtime qualification.

## Add an addon

1. Implement the model delta under `src/molgap/` and add an `AddonContract` in
   `experiment_spec.py`. Declare configuration with `AddonConfigField` (frozen
   literals or bounded integers), applicable family versions, source module and
   zero-initialization contract. Configuration validation uses these fields;
   it does not require an addon-name branch in the Spec parser.
2. Add the approved `TrainingAddon(name, version, mode, ...)` entries to the
   owning `TrainingAdapter` in `experiment_execution.py`. Generic graph addons
   bind `apply_hook="molgap.<module>:apply_addon"`; legacy adapters retain their
   mode dispatch. Reuse the family's or shared graph owner's recipe, preflight,
   training, output and resume hooks. A new dependency belongs in that
   descriptor; do not copy the full source inventory into an experiment.
3. Add focused checks for configuration, zero initialization, the owning mode
   and source dependency selection. No lifecycle, package, receipt, closure or
   platform launcher changes are needed for an existing-family addon.

Use `build_addon_declaration(family, addon, repo_root=..., config=..., version="1")`
to obtain the validated config and actual source digest. Use
`build_family_recipe(..., addon=..., addon_version="1")` for the recipe, then
freeze both in the Spec. `workflow-info --spec SPEC` reports the registered
execution support and preparation route. A declaration-only arm stays on its
owning `prepare-release` route; this query does not qualify execution.

Build the family's fixed recipe before freezing the Spec. This helper uses the
registered family builder; do not copy an old recipe JSON and edit constants.
The hashes come from the frozen training row and target identities:

```python
from molgap.experiment_execution import build_family_recipe

recipe = build_family_recipe(
    ("neural_atom_k1", "2"), addon="k1_joint_aggregation",
    source_idx_sha256="<frozen-source-index-sha256>",
    target_sha256="<frozen-target-sha256>",
)
```

Write the returned mapping as the canonical recipe file and pin its SHA256 in
`Spec.arms[].training.recipe`. For a reference arm, pass `addon=None`. The
helper accepts only registered families/addons and does not read development
rows. `k1-screen-v1` is the new adapter for `neural_atom_k1/2`; it is separate
from the retained `k1-v1` output protocol for `neural_atom_k1/1`.

For the current K1 and GPTrans candidate directions, the acceptance plan binds
the retained family reference by default. Add a baseline training arm only when
the owning contract explicitly requires reference retraining; neither the
workflow nor a recipe silently recreates one.

## Prepare once

First follow `AGENTS.md`, `BRANCHES.md`, the owning experiment contract, and
RML navigation. For a new question, freeze its prospective plan and acceptance
inputs before diagnostics or training. The exact Spec, prospective and
acceptance schemas remain with their owners; see [Spec](EXPERIMENT_SPEC.md),
[prospective planning](EXPERIMENT_PROSPECTIVE_PLANNING.md), and
[family output acceptance](EXPERIMENT_FAMILY_WORKFLOW.md).

The workflow plan is ordinary UTF-8 JSON with exactly these top-level fields:
`format`, `spec_identity`, `source_files`, `arms`, `acceptance_plan`, `kaggle`.
The current static platform registry accepts only the `kaggle` platform key;
this does not claim that the same entry point supports IMS or SCNet. `source_files`
adds repository-relative POSIX extras to the reviewed bootstrap and family
registrations; arm recipe paths are included automatically. Every Spec arm
appears exactly once. `recipe` is a repository-relative POSIX path pinned by
that arm's Spec; `initial_state` is an existing local file pinned by the Spec;
`device` is a distinct integer in the declared device range. `acceptance_plan`
points to the full pinned family acceptance plan under the repository root.

The `kaggle` object has exactly `account`, `kernel`, `title`, `datasets`,
`source_dataset`, and `accelerator`. The kernel must equal the account plus the
title-derived slug; title length is 5–50 characters. Dataset slugs must be
unique and include the account-owned source dataset. This workflow supports
the explicit `NvidiaTeslaT4` allocation only.

```json
{
  "format": "molgap-experiment-workflow-v1",
  "spec_identity": "<identity returned by validate-spec>",
  "source_files": ["src/molgap/<explicit-extra-module>.py"],
  "arms": [{
    "arm_id": "<Spec arm id>",
    "device": 0,
    "recipe": "experiments/<question>/training_contract.json",
    "initial_state": "D:/frozen/<arm>-initial-state.pt"
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

```powershell
$repoRoot = (Get-Location).Path
$python = '.\.venv\Scripts\python.exe'
& $python -m molgap.experiment_cli validate-spec --spec $spec
& $python -m molgap.experiment_cli prepare-workflow --spec $spec --repo-root $repoRoot --plan $plan --output $freshOutput
```

The prepared payload is not a submission. Follow the `kaggle-molgap-workloads`
skill and the [Kaggle adapter guide](../../platforms/kaggle/README.md) to publish
the source dataset and kernel. Pass `$freshOutput/release_report.json` to the
adapter's `--release-report` gate; it rechecks the report immediately before
POST. The platform skill owns submission and reconciliation of the same remote
attempt. Preserve its observed account, run reference and version in the
launch receipt. Do not treat a local receipt or queue response as acceptance.

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
changes during preparation. Restoration verifies the copied bytes against the
manifest before atomic publication and rejects concurrent destination conflicts.

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
