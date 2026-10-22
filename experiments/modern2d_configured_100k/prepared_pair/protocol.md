# Prepared PPGT / ESA execution screen

Authorized on 2026-10-22: one fresh Kaggle3 T4x2 allocation, at most 32400
seconds including installation, qualification, training and publication. Both
allocated devices count: at most 18 physical T4-hours, separately from at most9
Kaggle session quota hours. No automatic continuation. Kaggle3's SDK quota
observation showed12.3835 available session hours before preparation.

## Question and reuse

Can the two CPU-verified prepared execution paths pass native update/resume
qualification and complete useful fixed100K training within the wall budget?
The existing compiler, FamilyKernel/Session, indexed loaders, profiling,
checkpoint/output owners, journaled Kaggle adapter and terminal RML pipeline
remain the owners. No new training loop or graph cache is introduced.

One isolated GPU/RNG/optimizer/checkpoint directory per arm:

- PPGT/3: frozen compact initialization, existing mean-readout addon,
  lr0.001, weight_decay1e-5, prepared-v2 execution.
- ESA/4: frozen native initialization, no addon, lr0.0003,
  weight_decay1e-5, prepared-v2 execution.

The previous factorial screen supported mean readout for PPGT, but was budget
truncated; ESA residual initialization did not improve consistently across
learning rates. These choices avoid introducing another scientific mechanism.
Prior evidence: [campaign analysis](../research_arms/STATUS.md) and
[execution optimization verification](../../modern2d_model_integration/execution_optimization_2026-10-22.md).

## Frozen scope

Accepted indexed UnifiedV2 train100K and CommonDev50K, seed42, physical BS128,
FP32/TF32off, direct normalized Gap, no EMA. Each arm plans20 passes/15620
updates/1999360 sample presentations, with its frozen20-pass schedule.
No official validation, test, audit, geometry construction,500K or full run.
The declared authority is the only resource release; old campaign permission
and historical native certificates do not qualify the new implementation.

## Fail-closed execution gate

Before either arm trains, both run their normal train-only native repeat-a,
repeat-b and serialized-midpass-resume two-update probes. Additionally each
runs two updates through its existing accepted backend (PPGT bucketed-v1 or
ESA native), with identical initialization/data/recipe/RNG. Compare saved model
and Adam states with the existing tolerance, exact scheduler/cursor/RNG and
optimizer metadata; bind all source, runtime, recipe and state hashes. Any
failure stops the allocation, preserving proof files. Never fall back silently.
Two-update equivalence is not a claim of long-run numerical identity.

## Durability and decision

The normal atomic checkpoint cadence and independently retrievable chunks
remain enabled. The600-second closeout reserve is inside the9h wall ceiling.
Each arm captures a bounded profiler window, then detaches capture hooks.
Retain native proof, profile, curve/trace, predictions, selected/last states,
role events, actual allocation cost, terminal receipt and prospective binding.
If the horizon is incomplete, record STOP_FOR_COST/INCONCLUSIVE, not a win.
Missing historical references or costs stay explicit; do not manufacture
Replay-Ready or compare different families as a strict causal intervention.

The existing server Luna B observes only this exact bound job every30 minutes:
healthy states silent, UNKNOWN distinct from failure, terminal event once to
the existing controller, no automatic retries, modifications or successors.
