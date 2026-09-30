# Standard variant workflow

Use this for a new GPTrans-T or EdgeState-family variant on server. It composes
existing components; it does not introduce a trainer, submitter or evidence
schema. A small scientific change should not require another packager, account
wrapper, monitor or RML finalizer.

## What changes per experiment

Keep the addon implementation, reviewed registry entry, executable scientific
contract, comparison/reference decision, budget, and per-arm prospective plan
inputs beside the owning question. Freeze their actual identities in Spec v2.
Plan contents need scientific judgment: do not copy another experiment's
hypothesis, evidence IDs or authorization. Unchanged frozen references are
reused, not retrained to fill a device.

Everything else uses the following owners:

| Operation | Reusable entry | Input supplied by the experiment |
|---|---|---|
| Plan + package + upload layout + release check | `experiment_cli prepare-release` | Spec v2 and release workflow configuration |
| GPTrans arm arguments | `gptrans_screen_adapter.gptrans_screen_arguments` | Actual source package, fixed cache, initialization, transform and optional accepted sidecar |
| EdgeState arm context | `edge_state_screen_adapter.bind_edge_state_screen` | Actual package, pinned metadata, accepted runtime, train rows and approved statistics |
| Remote submit/reconcile/retrieve | Owning platform adapter and workload skill | Account, budget/role authority, staged inputs and returned physical identity |
| Monitoring | Existing server A/B binding | Exact returned job/version; no new conversation or cron |
| New-protocol output events, inspection and terminal translation | [Family output workflow](EXPERIMENT_FAMILY_WORKFLOW.md) | One-time owning-trainer hooks, frozen metric/exposure requirements and observed launch receipt |
| Saved-artifact acceptance | Owning family/experiment acceptance | Actual retrieved artifacts and frozen comparison conditions |
| RML closure | `experiment_cli terminal --execute` | Independently accepted per-arm terminal descriptor |

Independent acceptance is not replaced by a generic `accepted=true` flag.
The terminal descriptor delegates to existing RML wiring; roles, costs,
predictions, traces and reference binding must actually be present. Rebuild and
validate RML using its existing commands; inspect replay eligibility rather
than assuming every terminal entry is eligible.
Supported K1/GPTrans server screens can opt into the shared trainer event hooks;
use the family guide's exact capability limits. Science/cache/runtime/budget
approval and platform submission remain explicit owning stages, not CLI side
effects. Terminal closure can compose the replay-pair check with
`close_family_replay` without upgrading incomplete records.

## Local preparation configuration

`prepare-release` accepts `--spec`, `--workflow`, `--repo-root`, and a fresh
`--output`. The workflow JSON has exactly these fields:

```json
{
  "format": "molgap-release-workflow-v1",
  "spec_identity": "<actual Spec SHA256>",
  "source_paths": ["<explicit committed source allowlist>"],
  "artifacts": {
    "initial_state.pt": {"path": "<trusted local input>", "sha256": "<file SHA256>"}
  },
  "recipe_files": {"<arm_id>": "<packaged recipe path>"},
  "initial_states": {"<arm_id>": "initial_state.pt"},
  "required_modules": ["<selected family loader/trainer>"],
  "entry_template": "<committed template path>",
  "kernel_metadata": "<committed kernel metadata path>",
  "dataset_metadata": null,
  "pickle_inputs": []
}
```

This is field documentation, not runnable placeholder evidence. Paths are
explicit local paths (relative paths resolve under `--repo-root`). The entry
template and kernel metadata must be in the source allowlist. The template
contains exactly one `__PIN_SOURCE_ARCHIVE_SHA256__`; staging renders immutable
archive bytes and never relies on a mutable working copy after packaging.
Artifacts carry file SHA256 independently of the Spec's tensor-state SHA256.
Initialization bindings name staged artifacts, not arbitrary paths outside the
upload directory. Directory caches stay on accepted platform mounts; they are
not silently included as source or rebuilt on a GPU.

```powershell
.venv\Scripts\python.exe -m molgap.experiment_cli prepare-release --spec experiments/question/spec.json --workflow experiments/question/release_workflow.json --repo-root . --output platforms/_records/kaggle/packages/question_attempt_v1
```

The command first publishes per-arm plans through `plan_prospective`, then uses
`stage_release_inputs`. Its `workflow.json` retains both results. Upload inputs,
kernel files, source package and `release.json` are under `release/`.
The owning platform adapter still rechecks the release report before POST;
scientific prelaunch and runtime qualification remain separate gates.
No account selection or platform API call occurs in this command.

Partial planning or staging failure retains records and diagnostics. Do not
delete them, retry blindly, or hand-edit canonical RML. Reconcile the retained
plans before a new attempt; source changes require a new frozen source binding.

## Family wiring boundaries

GPTrans's argument adapter supports the V5 `degree_scale` and `path_bond_mean`
input arms. It supplies separate trajectory/run IDs and invokes no trainer.
Call the existing `run_preflight`/`run_training` in an already released isolated
worker, supplying output and preflight paths there. Construction registration
alone does not qualify other variants for V5 trace wiring. Persist Spec/arm
identity and verify it on resume; the argument adapter is not a resume gate.

EdgeState's context supports the registered base, depth-only and K1 construction
paths. It binds BS128, separate output/checkpoint/trace locations and delegates
sampler, model construction, step, evaluation and checkpoint/resume to
`edge_state_training_core`. Use its `recorder` with the executable contract's
explicit metric and device-time semantics; append only measured observations.
The context pins actual graph/contract metadata bytes, but does not prove role
acceptance, target statistics or runtime qualification. The experiment still
owns the frozen epoch loop, optimizer/scheduler, stopping and selected weights.
See [EdgeState boundaries](EDGE_STATE_ADAPTER.md). The model-only recipe is not
a training contract and cannot replace historical K1 or official EdgeState
implementations merely because their family names are similar.

CPU sidecar preparation/acceptance remains optional and mechanism-owned. Reuse
an already accepted sidecar; add a CPU builder only if the new mechanism needs
new input information. Neither a shared helper nor a full idle allocation
authorizes extra candidates, seeds or protected-role use.

## Verification boundary

Focused tests use temporary source repositories, small trusted tensors, metadata
and mocked family dispatch. They do not train, infer, access protected roles,
publish prospective experiments in this repository, or contact a platform.
No measured five-minute end-to-end submission claim follows from these tests.
