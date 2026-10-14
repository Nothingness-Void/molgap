# Shared 100K Operator Profile

Read [protocol](protocol.md) and [selection review](evidence_review.md).
The user authorized a Kaggle1 T4x2 full100K run and profiler capture on
2026-10-14, followed by desktop shutdown after verified submission.

Owning branch: `codex/exp/k1-gptrans-profiler-100k`.
Shared `FamilyTrainingSession` owns both arms. Platform operations use the
Kaggle skill, `run_experiment.py` and the accelerator API adapter.
[Config](config.json) owns training/resource dimensions; `prepare.py` only
declares/stages them through existing preparation APIs. No local trigger.

Reconcile the exact returned kernel/version and durable outputs in `submission/`
before choosing another action. The training decision stays pending until
native qualification, artifacts, traces, roles/cost and profiling are accepted.
Source/profile identity does not release promotion or protected roles.

The completed run's endpoints, mechanical/native checks and interpretation are
in the [dated terminal decision](terminal_acceptance/decision.md), with
[operator attribution](terminal_acceptance/attribution.md). RML/Git closure is
separate; the original prospective and submission records remain unchanged.
The [terminal custody record](terminal_acceptance/closure.md) owns the two
finalized, unpaired RML traces; [topology reuse](terminal_acceptance/execution_optimization.md)
owns subsequent shared-code changes and their separate qualification boundary.
