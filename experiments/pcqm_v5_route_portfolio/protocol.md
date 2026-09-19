# Protocol: V5 route portfolio

## Resource boundary

- Kaggle2 and Kaggle3 share a maximum budget of 60 GPU-hours for the week.
- Kaggle1 is reserved for desktop and must not be queried, submitted to,
  monitored, cancelled, or modified by this portfolio.
- Official validation, test-dev, test-challenge, shadow and desktop-owned jobs
  remain untouched.

## Stage 0 — prospective repeatability and reference calibration

Kaggle2 and Kaggle3 each run one independent repeat of the exact fixed
PCQM-100K K1-v4 contract: seed 42, deterministic FP32/no-TF32, physical BS128,
identical data/row order/features/target transform, optimizer, schedule, loss,
selection rule and sample exposure. These are one-time calibration runs, not a
baseline retrained per candidate.

The original accepted K1-v4 reference and the two repeats define a
prospectively documented training-run spread before either candidate result is
observed. Stage 0 must produce a reusable repository-local reference bundle,
portable target-transform asset, aligned predictions/rows/targets, runtime
certificate, trace, role events, native cost, acceptance, decision and RML
records. A failed calibration blocks candidate release rather than lowering a
gate.

## Stage 1 — two independent architecture questions

After Stage 0 acceptance, Route A and Route B may use Kaggle2/Kaggle3 in
parallel. Each route is a separate `architecture_comparison` with one declared
mechanism, one visible accelerator per independent model, seed 42, the fixed
PCQM-100K V5 contract, and the immutable Stage-0 reference. The baseline is not
retrained in candidate jobs.

Every route must freeze source/config first and then create, before submission:

1. a prospective `trajectory.json` with hypothesis, alternatives, related
   closed families, cheapest falsifier and expected native cost;
2. `training_contract.json`, role snapshot, budget snapshot and immutable cache
   acceptance;
3. a validated reference bundle and `comparison_readiness_prelaunch.json`;
4. runtime/preflight evidence including parameter count, memory and one
   optimizer-inclusive step;
5. atomic checkpoint, resume, trace, role-event and cost-event outputs.

Terminal collection must add aligned prediction/row/target manifests,
checkpoint/source/runtime hashes, observed `comparison_readiness.json`, paired
analysis, acceptance, decision and the trajectory result before RML rebuild.

## Stage 2 — attribution before another route

A negative result closes its exact mechanism and receives subgroup/trajectory
attribution before another submission. A positive result must exceed the
Stage-0 materiality rule and have a favorable paired interval. It earns only a
shortlist decision; it does not automatically authorize another seed, 500K,
full training or a protected role.

Unused GPU-hours remain unspent until Stage-1 evidence identifies one distinct
next bottleneck. No third architecture is pre-authorized merely to consume the
weekly quota.
