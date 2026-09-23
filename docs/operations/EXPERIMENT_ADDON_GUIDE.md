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

## Minimal Handoff Sequence

1. Follow `AGENTS.md`, `BRANCHES.md`, `CURRENT_STATE.md`, and the owning
   experiment contract. Query RML before opening a new research question.
2. Build and `validate-spec` a Spec v2 with exactly one arm-to-trajectory
   mapping per independent arm. Supply the frozen, SHA-pinned RML plan inputs;
   do not invent missing evidence.
3. Run `plan-prospective`, then package the explicit source set. A nonzero
   planning result may retain published trajectories: reconcile them before
   retrying. Use only a preflight mode supported by that family, or implement a
   scoped addon diagnostic without relabeling it as shared-core acceptance.
4. Implement the authorized trainer/platform addon and its tests. Follow
   `platforms/README.md`, `platforms/REMOTE_HANDOFF.md`, and the specific
   platform instructions before touching remote resources.
5. Submit only under the owning experiment's explicit resource/role authority.
   Reconcile the authoritative platform state and durable artifacts before
   creating terminal evidence or considering a retry.

The shared code makes identities and evidence flows reusable. It does not make
an experiment scientifically valid, authorize resource use, submit a job, or
promote a candidate.
