# Saved K1 diagnostic acceptance - 2026-10-08

## Scope and authorities

User-authorized saved500K trace and retained internal-development50K prediction
arithmetic only. No training, model execution, calibration, remote operation or
production adoption. These are NO_TRAIN diagnostics, not training replay-ready.

- [Trace decision](../../experiments/pcqm_k1_consistency_stage_analysis/attempt_002/terminal_decision.md)
  owns partial23/60 paired-stage interpretation and the pending terminal discriminator.
- [BN decision](../../experiments/pcqm_k1_bn_row_attribution/terminal_decision.md)
  owns posthoc row analysis and its selection/causality limitations.
- [First attempt](../../experiments/pcqm_k1_consistency_stage_analysis/terminal_decision.md)
  preserves the failed overly strict recipe-binding guard; its cost stays unknown.
- [Source custody](../../experiments/pcqm_k1_consistency_stage_analysis/attempt_002/source_custody.json)
  records the wrapper inventory-filter correction without changing frozen receipts.

## Engineering verification

The existing RML planner/finalizer/validator, atomic writers and paired-bootstrap
owner were reused. A saved-arithmetic adapter retains exact executed commit blobs;
the legacy frozen-inference closure remains byte-identical to desktop base
`ae0a9738147514b30210453f27df2b63a075f578`. Completed closure retries returned
ALREADY_FINALIZED without rewriting receipts. Existing policy semantics cannot
be overwritten by another preparation attempt.

Targeted pytest: **126 passed**, one dependency deprecation warning:
`test_k1_saved_analysis`, `test_k1_component_diagnostic`,
`test_k1_weight_average_diagnostic`, `test_k1_bn_calibration`,
`test_research_memory`, `test_v5_common`, `test_documentation_navigation`.
An earlier run found three navigation-length failures; root summaries were
compacted to pointers rather than loosening limits. `check --frozen` passed.
Committed-HEAD `check --frozen --portable` is an integration delivery gate.

Existing ignored RML dependencies were copied byte-for-byte for hash checking,
not decoded or used for new metrics. The only decoded prediction inputs were
the authorized internal-development50K tensors. New `results/row_deltas.pt`
is ignored, retained in the experiment checkout and desktop checkout, with
SHA256 `15796abd2d090296e07d720c2cdd8ddcb514f5b7fbdc73bccc4bc7e76b03dcef`.
An ordinary Git clone does not include ignored binaries; local retention is not
a claim that Git alone redistributes them. No historic replay qualification,
scientific conclusion, role authority or resource grant was upgraded.
