# Frozen GPTrans EMA portability

Owner: server / Track C. This experiment tests whether the accepted G1 EMA
advantage survives a disjoint fixed500K development cohort without retraining.
Scientific rationale and retained input locators are in the
[planning review](../pcqm_v5_route_portfolio/overnight_plan_2026-10-04.md).

The [contract](contract.json) owns execution/nomination thresholds;
[role plan](role_plan.json) and [budget](budget.json) bind scope and cost.
`run.py` is a thin platform bootstrap for `molgap.gptrans_portability`.
Two isolated T4 workers must reproduce both original 50K payloads before either
opens fixed500K development. No optimizer, training loader, EMA update or
checkpoint selection is executed. Geometry is never passed to the encoder.

The training-only ExperimentSpec registry cannot truthfully encode this audit.
The source archive therefore reuses `build_v4_source_bundle` (a format/IO helper,
not a V4 scientific-contract claim). `frozen_inference_release` verifies its
source inventory, every retained input, portable transform, executable entry
and mounts. The Kaggle adapter rechecks this report immediately before POST.
Remote deterministic reproduction is the runtime gate; local syntax or release
input qualification is not scientific acceptance.

Acceptance loads saved prediction tensors only and uses the existing paired
analysis. This NO_TRAIN record cannot be a new canonical training Replay pair
or STRICT_CAUSAL. Its two source training comparisons keep their own status.
The reused500K role is development, not a new sealed/generalization test.

After acceptance the existing controller may implement one evidence-justified
bounded training contrast under the user's authorization. Luna only hands off
terminal/fault evidence; it never trains/retries/selects a successor.

## Retained attempt decisions

- [v1 installation failure](results/failure_v1/decision.md).
- [CPU interpreter qualification](results/environment_v1/decision.md).
- [v2 mount-resolution failure](results/failure_v2/decision.md).
- [v3 role-event serialization failure](results/failure_v3/decision.md).
- [v4 accepted frozen portability](attempt_v4/decision.md).
- [500K training capability/release review](scale_training_review.md).
