# Experiment Addon Guide

This is the short handoff for extending the shared experiment core. It defines
ownership boundaries; the linked contracts remain authoritative for field-level
schemas and safety checks.

## Shared Core

Reuse these modules instead of copying their logic into an experiment script:

- `experiment_spec.py`: strict immutable experiment identity and arm bindings.
- `experiment_prospective.py` and `research_memory/plan.py`: per-arm prospective
  RML planning against one frozen evidence snapshot.
- `experiment_package.py`: explicit allowlist, source archive and package hash.
- `experiment_preflight.py`: only the family/mode combinations documented in
  `EXPERIMENT_PREFLIGHT.md`; unsupported means unsupported, not an implicit pass.
- `experiment_runner.py`: metadata probe and model construction diagnostics
  only; it does not train or load a checkpoint.
- `experiment_launch.py`: local immutable launch/reconciliation receipts only;
  it does not contact a platform.
- `experiment_terminal.py`: validates terminal bindings and delegates explicit
  closure to the existing RML pipeline.

Start with the [local CLI](EXPERIMENT_CLI.md), then read only the relevant
detailed contract: [spec](EXPERIMENT_SPEC.md),
[prospective planning](EXPERIMENT_PROSPECTIVE_PLANNING.md),
[package](EXPERIMENT_PACKAGE.md), [preflight](EXPERIMENT_PREFLIGHT.md),
[runner](EXPERIMENT_RUNNER.md), or [launch receipts](EXPERIMENT_LAUNCH.md).

## Addon Ownership

The experiment-owned addon is responsible for family-specific training and
checkpoint/resume semantics, any extra model/data validation not supported by
the shared preflight, and platform-specific submission, status reconciliation
and artifact retrieval. Prefer a thin wrapper around existing training code.
Do not copy reusable model, RML, source-packaging or receipt logic into a
second implementation.

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
   experiment contract. Query RML before opening a new research question.
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
5. Use the owning trainer/platform addon and its tests. Follow
   `platforms/README.md`, `platforms/REMOTE_HANDOFF.md`, and the specific
   platform instructions before touching remote resources.
6. Submit only under the owning experiment's explicit resource/role authority.
   Reconcile the authoritative platform state and durable artifacts before
   creating terminal evidence or considering a retry.

The shared code makes identities and evidence flows reusable. It does not make
an experiment scientifically valid, authorize resource use, submit a job, or
promote a candidate.
