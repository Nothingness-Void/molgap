# Modular Experiment Workflow

This is the entry point for the registered local-to-Kaggle training path. It
owns preparation and acceptance orchestration. It does not own scientific
authorization, a trainer loop, or platform submission.

## Owners

| Layer | Owner | Responsibility |
|---|---|---|
| Lifecycle | `experiment_workflow.py`; `prepare-workflow` / `accept-workflow` in the local CLI | Reuse package, release-check, prospective-planning, family-output inspection and existing terminal closure owners. |
| Family execution | `experiment_execution.py`; `experiment_training_worker.py` | Static family/addon-to-trainer dispatch and one isolated subprocess per arm/phase. No caller-selected import or callback. |
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
| `neural_atom_k1/2` | `reference` | `k1_joint_aggregation/1` → `ssma`; `k1_two_pass_mean/1` → `mean2`; `k1_mean2_clean_second/1` → `mean2_clean_second` | `k1-screen-v1` |
| `gptrans_t/1` | `reference` | `pair_prenorm/1`, `centered_logits/1`, `memory_value/1`, `memory_message/1` | `gptrans-v1` |

The executable registry is in `experiment_execution.py`; the output profiles
are in `experiment_family_artifacts.py`. `experiment_source_inventory.py`
owns reviewed shared bootstrap files, while each family adapter declares its
serialized-module dependencies in the execution registry. The family trainer owns its model,
loader, optimizer, selection and resume behavior. A same-family variant adds a
registered addon/mode once; each experiment then supplies only that addon,
its frozen contract/recipe and config. It does not add an experiment-specific
training loop or launcher. A new family registers its declarative family
contract, model/trainer adapter, output profile and reviewed source inventory
once. A new addon within an existing family adds its reviewed mode hook and
source path only when it introduces a new dependency. Unsupported families,
addon versions, modes and platform allocations stop before execution.

`mean2` changes only the training loss to the mean of two stochastic-forward
L1 losses, without consistency penalty, teacher or EMA. Evaluation remains a
single clean forward. Its intentional two-forward preflight cost ceiling is
100% overhead versus single-forward; SSMA keeps its 25% ceiling. Neither is a
scientific promotion rule. Scientific role (`reference`/`candidate`) is distinct
from executable mode.

`mean2_clean_second` keeps two equal supervised losses, one optimizer update
and two training-mode BN updates. Only the second forward disables K1 dropout,
including functional LocalGPSBlock and attention dropout. It is not consistency
regularization, clean-evaluation BN calibration, EMA or an inference ensemble.

K1 recipe construction accepts an explicit `seed` and `initialization_sha256`.
Seed42 defaults retain their historical identity. Another seed requires a fresh
CPU tensor artifact; both arms load its full pinned state, rather than expecting
different runtime versions to reproduce random construction. The sampler and
worker hash seed bind the declared seed. This interface grants no new seed run.

Commit executable source and recipes first. Generate prospective actions against
that exact HEAD, then prepare without another source commit in between. The
workflow rejects source/action/run/attempt mismatches before publication:
training traces use `logical_run_id:arm_id:downstream`; a same-job pair uses
one shared `logical_run_id-vN` attempt, with N later verified from the launch
response. It never rewrites old plans to fit observed execution.

A recipe may additionally pin `allocation_wall_limit_seconds` (120--14400).
Every arm must agree. The Kaggle bootstrap subtracts installation/setup time
and reserves 60 seconds for cleanup before passing the remaining budget to the
shared pair runner. Timeout records `STOP_FOR_COST`, retains the last atomic
complete-epoch checkpoint, and cannot claim the planned endpoint. Historical
recipes without this optional field keep their existing execution behavior.

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

`prepare-workflow` requires a fresh output directory outside `experiments/`;
use ignored `platforms/_records/<platform>/staging/` so staged trajectory copies
cannot be rediscovered as canonical RML. In a new worktree, verify existing RML
and materialize its declared retained artifacts before preparation; Git does
not copy ignored artifacts between checkouts. It checks every arm and
the pinned acceptance inputs, builds the immutable source package, runs the
family's static recipe checks, and runs `check_release_inputs`. These checks
bind the fixed recipe, initialization, sampler, roles, exposure and packaged
bootstrap before prospective publication.
K1 v2 release also reads the staged initial artifact through the frozen package's
actual family loader in a CPU-only subprocess. Both flat and `model_state`
envelopes share the validated reader; generic tensor inspection alone is not
transport compatibility. This check constructs no model and grants no GPU release.
The prospective planner validates all arms before publishing. The workflow then
freezes the launch config and
runs `check_workflow_binding` over the config, staged dataset mounts, T4 and
device shape, kernel metadata, prospective trajectory hashes and final entry
script hash. `release_report.json` records this final binding; the Kaggle
adapter rejects a missing, failed or stale report and repeats the byte checks
before POST. The output contains `package/`, `source_dataset/`, `kernel/`,
`release_report.json`, and `workflow_report.json`.

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

## Evidence boundary

Package/release checks and synthetic local fixtures establish interface and
mechanical behavior only. They do not establish real-shard execution, GPU
qualification, scientific validity or training gains. The platform skill owns
remote state and retrieval; the family contract and RML/V5 owners govern
acceptance. See [architecture](../../ARCHITECTURE.md) and
[family output checks](EXPERIMENT_FAMILY_WORKFLOW.md).
