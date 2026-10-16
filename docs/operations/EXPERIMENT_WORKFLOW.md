# Server variant API compatibility

Compatibility/API reference: new work starts with the
[quickstart](EXPERIMENT_QUICKSTART.md), one capability catalog and the fixed
runtime distribution. The retained `prepare-release` APIs below keep their
scientific and custody boundaries; they are not another default workflow.

This page retains the server GPTrans/EdgeState release APIs and strict Replay
boundary. It does not select new work or upgrade historical family versions.

## Registered family training workflow

Query `catalog` / `workflow-info --spec SPEC` for capabilities. The
[registered API](REGISTERED_EXPERIMENT_WORKFLOW.md) owns `prepare-workflow`,
acceptance and recovery. Existing author/input screens and EdgeState event hooks
retain their adapters; `prepare-release`/`check-preparation` below remain callable.
Registration and local tests do not release compute or native qualification.

## What changes per experiment

The question owns its scientific delta, comparison/reference, authority and
prospective inputs. The [addon API](EXPERIMENT_ADDON_GUIDE.md) owns extension
registration; the [architecture map](../../ARCHITECTURE.md) locates lifecycle
owners. Reuse unchanged references. The
[family output API](EXPERIMENT_FAMILY_WORKFLOW.md) owns event hooks, inspection
and terminal translation; `close_family_replay` composes the existing pair
check. Actual roles, costs, predictions, traces and accepted reference bindings
remain required. Server A/B monitors only exact server-owned jobs; shared code
does not adopt Desktop jobs or release full training/evaluation.

### Replay-first release and closure

For a server training screen, submission requires a Replay-capable **planned**
closure, not a promise that a future job already has terminal evidence. Before
releasing compute, freeze the exact reference bundle and verify its accepted
canonical reference trace, own evidence identity, enrollment eligibility,
distinct role/cost/acceptance pointers and retained artifact hashes. Use
`server_acceptance.validate_server_scientific_prelaunch` with the actual bundle
path/root; a valid JSON prelaunch alone does not release compute. The GPTrans
qualified reference has a hash-bound `qualification_ref`, reverified by this
release gate; the original shared-pointer bundle is rejected for new strict runs.

Each arm must plan the full live/EMA trace, cumulative steps/presentations,
atomic checkpoints, aligned predictions/rows/targets, measured native costs,
role history and independently accepted terminal decision. Preserve budget
estimates separately from incurred measurements. At closure, require the actual
candidate AND its frozen control in the rebuilt Replay pool, with a shared
comparability key and complete trace capability; do not infer success from
`accepted=true`, `strict_ready=true` or manifest eligibility alone. Failed or
incomplete jobs remain explicitly incomplete, never fabricated Replay-Ready.

Existing finalized records are immutable. A binding-only recovery can use
`research_memory.candidate_qualification` after verifying the original prospective
bundle and the reference's additive qualification prove identical scientific
and observed identities. It recomputes the strict assessor and is not a recipe,
checkpoint, gate or prospective-information replacement. G1/G2's owning wrapper
is `experiments/pcqm_gptrans_author_alignment/qualify_candidates.py`.

## Legacy release configuration

Direct legacy callers publish Specs through `ExperimentSpec(payload).write(path)`;
the [CLI byte contract](EXPERIMENT_CLI.md#input-and-output-contract) applies to
these frozen machine records. Normal editable inputs use the quickstart's
automatic freeze path and accept readable JSON.
For upload inputs, use `UploadArtifact.from_file(path).to_workflow()` when
freezing the workflow. This computes **file bytes SHA**, distinct from tensor
state SHA and target-transform semantic identity. Consumption verifies that
frozen SHA; it never silently refreshes a changed artifact to make a gate pass.

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
.venv\Scripts\python.exe platforms/local/run.py --profile $profilePath -- prepare-release --spec $spec --workflow $workflow --repo-root . --output $freshOutput
```

After read-only preparation checks, the command publishes per-arm plans through `plan_prospective`, then uses
`stage_release_inputs`. Its `workflow.json` retains both results. Upload inputs,
kernel files, source package and `release.json` are under `release/`.
The owning platform adapter still rechecks the release report before POST;
scientific prelaunch and runtime qualification remain separate gates.
No account selection or platform API call occurs in this command.

Before publication, `prepare-release` automatically executes the same
read-only checks available independently as:

`check-preparation --spec SPEC --workflow WORKFLOW --repo-root ROOT`.

The check validates every arm using the actual RML planner's schema, policy,
reference, closed-route context, pointer and cost rules without writing plans
or derived files. It also checks selected committed sources, syntax, required
module presence, LF-normalized recipe SHA, initialization bindings, entry pin,
and actual upload-file SHA. These failures occur before **any** arm is published.
It is not a replacement for real-reference scientific prelaunch, dependency
imports, finite tensor inspection, runtime qualification or platform recheck.
Run it separately only for a read-only check; do not routinely run it and then
repeat it via `prepare-release`.

`workflow.json.timings` measures preparation check, prospective publication,
combined package/stage/release check, and total local preparation with a monotonic
clock. Failures after publication retain elapsed phase time for reconciliation.
These measurements exclude addon design, platform upload/POST and monitor handoff;
they are not an end-to-end submission SLA or a measured five-minute claim.
Upload and remote restore use the source-package owner's complete sidecar list,
including `package_manifest.json`; the mount gate checks their exact bytes.

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

Focused CPU fixtures do not establish real-data/native qualification or measured
submission latency. Shared K1/GPTrans execution uses the
[training core](FAMILY_TRAINING_CORE.md); compatibility scope is retained in
[the integration record](SHARED_TRAINING_SERVER_INTEGRATION.md).
