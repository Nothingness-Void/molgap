# Experiment Addon Guide

This is the short handoff for extending the shared experiment core. It defines
ownership boundaries; the linked contracts remain authoritative for field-level
schemas and safety checks.

## Pick the Operation

Select the existing owner before adding plumbing. Paths below are reuse
entry points, not universal APIs or permission to execute a closed experiment.
Read the owning contract and inspect the callable's signature and nearest
caller/test; direct library reuse is valid without the shared CLI.

| Operation | Reuse owner | Boundary / next pointer |
|---|---|---|
| Train GPTrans | [pcqm_gptrans_v4.py](../../src/molgap/pcqm_gptrans_v4.py): `run_training`; [pcqm_gptrans_full_runner.py](../../src/molgap/pcqm_gptrans_full_runner.py): `train_full` | Fixed V4 screen and full-role recipes are distinct. Start at the [V4 reference](../../experiments/pcqm_gptrans_t_100k_v4/README.md) or [full-run contract](../../experiments/pcqm_k1_gptrans_full_fusion/README.md), not a copied launcher. |
| Train EdgeState | [edge_state_training_core.py](../../src/molgap/edge_state_training_core.py): statistics, sampler, step, evaluation, `save_checkpoint` / `restore_checkpoint`; [pcqm_official_edge_state.py](../../src/molgap/pcqm_official_edge_state.py): `train_official_edge_state` | The Spec-bound core supplies primitives, not a complete recipe. The official trainer has its [own experiment](../../experiments/pcqm_edge_state_full/README.md); the [model-only adapter](EDGE_STATE_ADAPTER.md) is not a trainer. |
| Resume or infer from a checkpoint | Matching family trainer's resume loader and model factory; public [inference.py](../../src/molgap/inference.py), lazily exported by [__init__.py](../../src/molgap/__init__.py) | Use the accepted checkpoint's owner and [asset map](../../models/README.md), not an arbitrary constructor. GPTrans full scoring uses [pcqm_gptrans_official_eval.py](../../src/molgap/pcqm_gptrans_official_eval.py); public loaders are not universal Track B adapters. |
| Analyze saved predictions | Owning artifact/row-alignment analysis; [analyze_pair.py](../../experiments/pcqm_gptrans_100k_transfer_control/analyze_pair.py): `paired_metrics`, tested by [test_gptrans_local_pair_analysis.py](../../tests/test_gptrans_local_pair_analysis.py) | This callable compares retained aligned rows/targets without model execution. The script's full CLI also performs checkpoint inference: do not run it as prediction-only analysis. Match identities and roles before reusing the example; row bootstrap is not training variance. |
| Bind source and local observations | [local CLI](EXPERIMENT_CLI.md) and the shared core below | Spec, package, preflight and receipts do not train or submit. `run-diagnostic` probes metadata or constructs a random model; it does not load checkpoints or run inference. |
| Accept artifacts and qualify a comparison | Owning family acceptance; [family output hooks](EXPERIMENT_FAMILY_WORKFLOW.md); [pcqm_gptrans_full_acceptance.py](../../src/molgap/pcqm_gptrans_full_acceptance.py): `accept`; [comparison_readiness.py](../../src/molgap/comparison_readiness.py) | Output hooks support only their capability matrix. V5 readiness does not generate predictions. Mechanical acceptance, scientific decision, runtime qualification and promotion remain separate under the [V5 contract](MOLGAP_COMMON_DIRECTION_V5_FINAL.md). |
| Close and index evidence | [RML entry point](../../research_memory/README.md), [lifecycle](../../research_memory/LIFECYCLE.md), [experiment_terminal.py](../../src/molgap/experiment_terminal.py), [terminal_wiring.py](../../src/molgap/research_memory/terminal_wiring.py) | Query RML first and follow canonical evidence. Terminal translation is read-only by default, not acceptance; explicit closure retains existing gates. Rebuild/check after accepted milestones, never hand-edit derived indexes. |

Platform submission, authoritative status reconciliation and retrieval stay in
the `kaggle-molgap-workloads`, `ims-molgap-workloads`, and
`scnet-bw-dcu-molgap` skills and their existing adapters. This guide does not
duplicate their remote commands; [platforms/README.md](../../platforms/README.md)
routes repository boundaries. A missing capability is a gap to report, not a
reason to alter a frozen recipe or silently retrain a reference.

## Shared Core

Reuse these modules instead of copying their logic into an experiment script:

- `experiment_spec.py`: strict immutable experiment identity and arm bindings.
- `experiment_prospective.py` and `research_memory/plan.py`: per-arm prospective
  RML planning against one frozen evidence snapshot.
- `experiment_package.py`: explicit allowlist, source archive and package hash.
- `experiment_preflight.py`: only the family/mode combinations documented in
  `EXPERIMENT_PREFLIGHT.md`; unsupported means unsupported, not an implicit pass.
  Its separate `check_release_inputs` checks declared source/recipe/import,
  serialized dependency and initialization bindings before platform publication.
- `experiment_runner.py`: metadata probe and model construction diagnostics
  only; it does not train or load a checkpoint.
- `experiment_launch.py`: local immutable launch/reconciliation receipts only;
  it does not contact a platform.
- `experiment_terminal.py`: validates terminal bindings and delegates explicit
  closure to the existing RML pipeline.
- `experiment_family_workflow.py` and `experiment_family_artifacts.py`: producer
  event hooks, frozen-contract output inspection and terminal translation;
  see the [family workflow](EXPERIMENT_FAMILY_WORKFLOW.md) capability matrix.

Start with the [local CLI](EXPERIMENT_CLI.md), then read only the relevant
detailed contract: [spec](EXPERIMENT_SPEC.md),
[prospective planning](EXPERIMENT_PROSPECTIVE_PLANNING.md),
[package](EXPERIMENT_PACKAGE.md), [preflight](EXPERIMENT_PREFLIGHT.md),
[runner](EXPERIMENT_RUNNER.md), or [launch receipts](EXPERIMENT_LAUNCH.md).
For an EdgeState baseline, depth change, or K1 architecture comparison, start
with [EDGE_STATE_ADAPTER.md](EDGE_STATE_ADAPTER.md); its model-only recipe is
not an executable training contract.

## Addon Ownership

The experiment-owned addon is responsible for its frozen scientific recipe,
authenticated graph/data loader, sampler and exact row-order replay, training
loop, selection, and any validation not supported by shared preflight. For an
EdgeState arm, call the shared training core rather than copying its step,
evaluation, or checkpoint code. Do not copy reusable model, RML,
source-packaging or receipt logic into a second implementation. Platform
submission, status reconciliation, and artifact retrieval belong to the
applicable Kaggle/IMS/SCNet skill and its existing platform adapter, not to
the shared CLI or experiment training core.

The current shared CLI intentionally has no training command or platform
submitter. The new addon must not claim a successful dry run is a submitted job,
or treat a queue response as completed execution. Use the owning platform's
handoff and access boundary. Credentials stay outside the repository.

`ExperimentSpec` uses reviewed static family/addon registries. A new family or
addon must add its small contract entry and focused tests; arbitrary import
strings, callbacks and runtime plugin discovery are deliberately unsupported.
The addon owns execution and platform behavior, while the shared core retains
canonical identity, path confinement, immutable packaging and evidence rules.

## Desktop arm packing

For desktop-owned accelerator submissions, plan two independent arms in one
physical job when both have an approved decision-relevant question and the
platform allocation makes the combined run fit its time and memory limits.
Prefer concurrent isolation on two assigned GPUs; on one GPU, use sequential
arms only when the combined frozen runtime and checkpoint plan fit the job
limit. Record the actual GPU assignment and native cost for each arm. Keep
separate prospective trajectories, arm identities, initialization, checkpoints,
traces, role-use, and terminal decisions even when they share one input dataset
and launch receipt. A failure in one arm must not silently supply evidence for
the other.

Check RML and the owning contracts before choosing the second arm. A frozen,
accepted reference is reused for comparison unless a new contract makes its
retraining scientifically necessary. Accelerator utilization alone does not
authorize a duplicate experiment, a second seed, a changed scientific recipe,
or protected-role access. If only one arm is justified or the combined run
cannot be made durable within the platform limit, submit the justified arm
alone and record why two arms were not feasible.

## Minimal Handoff Sequence

1. Follow `AGENTS.md`, `BRANCHES.md`, `CURRENT_STATE.md`, and the owning
   experiment contract. Query RML before opening a new research question. For
   a desktop-owned new experiment, create its dedicated branch and worktree from
   the verified current `molgap-desktop` tip before changing source or publishing
   prospective records. Complete training reconciliation, acceptance, decision,
   and terminal RML on that branch; route the closed result by `BRANCHES.md`.
2. Identify the existing family trainer and platform submitter before writing
   an addon. Inspect the platform input mounts, staged source shape, kernel
   entry point, and metadata limits with read-only or local static checks.
   Finish the executable bootstrap and source allowlist, then freeze the source
   commit. This inspection is not a diagnostic or training experiment.
3. Build and `validate-spec` a Spec v2 with exactly one arm-to-trajectory
   mapping per independent arm. Supply frozen, SHA-pinned RML plan inputs
   bound to the executable source commit; do not invent missing evidence.
4. Run `plan-prospective` before any new diagnostic or training experiment,
   then package the explicit source set, run `check-release`, and run the supported preflight. A
   nonzero planning result may retain published trajectories: reconcile them
   before retrying. If executable source changes before submission, reconcile
   the superseded plan and plan again against the new commit; never hand-edit
   canonical RML records. Use only a preflight mode supported by that family,
   or implement a scoped addon diagnostic without relabeling it as shared-core
   acceptance.
5. Use the owning trainer/platform addon and its tests. Follow
   `platforms/README.md`, `platforms/REMOTE_HANDOFF.md`, and the specific
   platform instructions before touching remote resources.
   For a trainer emitting the family output protocol, run `check-acceptance`
   before release to check frozen expectations and retained reference inputs.
   This availability check is separate from executable bootstrap preflight.
6. Submit only under the owning experiment's explicit resource/role authority.
   Reconcile the authoritative platform state and durable artifacts before
   creating terminal evidence or considering a retry.
7. After terminal acceptance, close the scientific question with the owning
   decision and a failure-mode attribution beside it before selecting another
   module. Apply `AGENTS.md`'s terminal attribution rule to accepted traces,
   paired artifacts, role use, and native cost. State the exact unresolved
   discriminator when evidence cannot identify a cause; do not use another
   architecture submission to substitute for this analysis. Rebuild and check
   RML for accepted canonical milestones, then route the closed branch by
   `BRANCHES.md`.

The shared code makes identities and evidence flows reusable. It does not make
an experiment scientifically valid, authorize resource use, submit a job, or
promote a candidate.
