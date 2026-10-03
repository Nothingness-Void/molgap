# Experiment Addon Guide

This is the short handoff for extending the shared experiment core. It defines
ownership boundaries; the linked contracts remain authoritative for field-level
schemas and safety checks.

New agents should begin with the [infrastructure quickstart](EXPERIMENT_QUICKSTART.md),
then return here for the addon ownership contract. The [registered workflow](REGISTERED_EXPERIMENT_WORKFLOW.md)
contains the executable registry and recovery details.

Before adding plumbing, use the repository's
[experiment reuse skill](../../.agents/skills/molgap-experiment-reuse/SKILL.md).
It routes to existing components and records only the missing behavior; the
contracts below remain authoritative. It does not add a second execution core.

## Shared Core

Reuse these modules instead of copying their logic into an experiment script:

- `experiment_cli.py`: one local entry point for the shared operations below;
  it has no platform submitter.
- `experiment_spec.py`: strict immutable experiment identity and arm bindings.
- `experiment_prospective.py` and `research_memory/plan.py`: per-arm prospective
  RML planning against one frozen evidence snapshot.
- `experiment_package.py`: explicit allowlist, source archive and package hash.
- `experiment_workflow.py` and `experiment_staging.py`: compose prospective
  planning, immutable upload layout and existing release checks. For GPTrans
  or EdgeState variants, start with [the standard workflow](EXPERIMENT_WORKFLOW.md).
- `experiment_preflight.py`: only the family/mode combinations documented in
  `EXPERIMENT_PREFLIGHT.md`; unsupported means unsupported, not an implicit pass.
  Its `check_release_inputs` checks selected source dependencies, packaged
  recipes, frozen initialization and upload bindings before platform publication.
- `experiment_runner.py`: metadata probe and model construction diagnostics
  only; it does not train or load a checkpoint.
- `experiment_launch.py`: local immutable launch/reconciliation receipts only;
  it does not contact a platform.
- `experiment_terminal.py`: validates terminal bindings and delegates explicit
  closure to the existing RML pipeline.
- `experiment_family_workflow.py` and `experiment_family_artifacts.py`: producer
  event hooks, frozen-contract output inspection and terminal translation;
  see the [family workflow](EXPERIMENT_FAMILY_WORKFLOW.md) capability matrix.
- `edge_state_training_core.py`: Spec-bound EdgeState target statistics,
  deterministic full-batch sampler, OGB batch checks, normalized Gap step,
  source-aligned development predictions, and SHA-pinned checkpoint/resume.

Start with the [local CLI](EXPERIMENT_CLI.md), then read only the relevant
detailed contract: [spec](EXPERIMENT_SPEC.md),
[prospective planning](EXPERIMENT_PROSPECTIVE_PLANNING.md),
[package](EXPERIMENT_PACKAGE.md), [preflight](EXPERIMENT_PREFLIGHT.md),
[runner](EXPERIMENT_RUNNER.md), or [launch receipts](EXPERIMENT_LAUNCH.md).
For an EdgeState baseline, depth change, or K1 architecture comparison, start
with [EDGE_STATE_ADAPTER.md](EDGE_STATE_ADAPTER.md); its model-only recipe is
not an executable training contract.

## Server capability boundary

| Task | Reuse | Remaining experiment responsibility |
|---|---|---|
| Identity and prospective records | Spec v2 and per-arm RML planning | Authenticated scientific, role, budget and source inputs |
| Source and receipts | Explicit source package and local launch receipts | Platform submission, verified responses and server A/B binding |
| Repeated variant preparation | `prepare-release` plus family screen adapters | Scientific delta, explicit per-arm plans and independently accepted inputs |
| Family diagnostics | Metadata probes and reviewed construction adapters | Supported-mode checks and actual accelerator qualification |
| EdgeState training primitives | Target statistics, sampler, step, evaluation and checkpoint helpers | Accepted graph loader, frozen training loop, selection, native cost and trace |
| New-protocol K1/GPTrans outputs | Family session, inspector and selective Kaggle retrieval | One-time owning-runner integration; frozen metric/endpoint requirements and actual receipts |
| Terminal closure | Descriptor translation and explicit RML execution | Independent acceptance, actual artifacts and eligible candidate/reference evidence |

Read [K1 adapter compatibility](K1_ADAPTER.md) before selecting its registered
addons. Registration or a successful metadata probe does not establish factory
compatibility. K1/EdgeState loader-only preflight checks selected frozen topology
shards, not a complete role or model. Model smoke remains unsupported for those
families; experiment-owned training admission remains separate.
Verification scope and unresolved integration gaps are recorded in
[the server integration review](shared_experiment_verification.md).

An existing immutable reference remains the default. Optional
`prospective.same_run_replay` is for an explicitly justified new paired run,
not a reason to retrain the baseline for every candidate. Merely writing a
terminal descriptor does not make either arm replay-ready.

## Addon Ownership

The addon owns only its model delta, typed configuration, and any contract
specific validation or hook that the selected family owner does not provide.
The family trainer or shared graph owner supplies its declared input loader,
sampler, exact row-order replay, training loop, selection, and resume behavior.
For an EdgeState arm, call the shared training core rather than copying its
step, evaluation, or checkpoint code. Do not copy reusable model, RML,
source-packaging or receipt logic into a second implementation. Platform
submission, status reconciliation, and artifact retrieval belong to the
applicable Kaggle/IMS/SCNet skill and its existing platform adapter, not to
the shared CLI or experiment training core.

The current shared CLI intentionally has no training command or platform
submitter. The new addon must not claim a successful dry run is a submitted job,
or treat a queue response as completed execution. Use the owning platform's
handoff and access boundary. Credentials stay outside the repository.

On `molgap-server`, prospective planning and terminal execution retain the
server RML's server-owned trajectory boundary. Desktop-owned records stay on
the independent desktop checkout; this shared code does not transfer custody.

`ExperimentSpec` uses reviewed static family/addon registries. A new family or
addon must add its small contract entry and focused tests; arbitrary import
strings, callbacks and runtime plugin discovery are deliberately unsupported.
The addon owns its model delta and scientific configuration. The family owner
implements execution hooks, the platform adapter implements platform behavior,
and the shared core retains canonical identity, path confinement, immutable
packaging and evidence rules.

For an addon within a registered training family, follow the
[three-step registration contract](REGISTERED_EXPERIMENT_WORKFLOW.md#add-an-addon).
Typed config fields and a small execution descriptor bind the model delta,
mode and extra dependencies. Generic graph addons additionally bind a reviewed
`apply_addon` hook whose source is hash-pinned; legacy family addons retain
their owning mode dispatch. The lifecycle, launch validation, recovery transport
and acceptance orchestration stay reusable. Platform operations remain with the
owning workload adapter.

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

1. Follow the selected checkout's `AGENTS.md`, `CURRENT_STATE.md`, and owning
   experiment contract. Query RML before opening a new research question.
   On server, branch governance is in `AGENTS.md`: use `molgap-server`, with
   temporary isolation only when needed. Desktop's branch/worktree policy
   belongs to its own checkout; do not import that workflow or adopt its jobs.
2. Identify the existing family trainer and platform submitter before writing
   an addon. Inspect the platform input mounts, staged source shape, kernel
   entry point, and metadata limits with read-only or local static checks.
   Finish the executable bootstrap and source allowlist, then freeze the source
   commit. This inspection is not a diagnostic or training experiment.
3. Build and `validate-spec` a Spec v2 with exactly one arm-to-trajectory
   mapping per independent arm. Supply frozen, SHA-pinned RML plan inputs
   bound to the executable source commit; do not invent missing evidence.
4. Run `plan-prospective` before any new diagnostic or training experiment,
   then package the explicit source set and run the supported preflight. A
   nonzero planning result may retain published trajectories: reconcile them
   before retrying. If executable source changes before submission, reconcile
   the superseded plan and plan again against the new commit; never hand-edit
   canonical RML records. Use only a preflight mode supported by that family,
   or implement a scoped addon diagnostic without relabeling it as shared-core
   acceptance.
5. Use the owning trainer/platform addon and its tests. Run
   `check-release` with the actual entry script and staged inputs for new
   ExperimentSpec source packages; retain the report for the platform adapter's
   pre-submission recheck. This does not replace the owning scientific gate.
   Follow `platforms/README.md`, `platforms/REMOTE_HANDOFF.md`, and the specific
   platform instructions before touching remote resources.
   For a trainer emitting the family output protocol, run `check-acceptance`
   before release to check frozen expectations and retained reference inputs.
   This availability check is separate from executable bootstrap preflight.
6. Submit only under the owning experiment's explicit resource/role authority.
   Reconcile the authoritative platform state and durable artifacts before
   creating terminal evidence or considering a retry.

The shared code makes identities and evidence flows reusable. It does not make
an experiment scientifically valid, authorize resource use, submit a job, or
promote a candidate.
