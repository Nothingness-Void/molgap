# Phase 8 Repair-v2 Pure 2D Gate Decision

## Decision

**Negative: stop repair-v2 before 3D graph construction, SchNet training, or
2D+3D fusion.** The coverage-repair mixture does not meet the fixed external
pure-2D gate. It is not a registry candidate and does not reopen the MoE path.

## Controlled setup

- v1 and v2 are independent 1M dual-GPS7/GPS9 late-fusion models.
- Both use seed 42, the same `DualGPSFusionHead` architecture, optimizer,
  uniform head sampling, and external 2D graph construction.
- v1 uses the rejected 1M continuation's GPS encoders and controlled head.
- v2 preserves the validated expansion500K prefix and replaces only its appended
  half with the coverage-repair top-up.
- All decisions below use paired per-molecule absolute-error deltas
  (`v2 - v1`) with 10,000 bootstrap draws. The two model-specific internal
  holdouts are descriptive only because their distributions are different.

## Fixed external results

| evaluation | n | average MAE delta (eV) | Gap MAE delta (eV) | Gap 95% CI (eV) | result |
|---|---:|---:|---:|---:|---|
| common all | 1,980 | +0.000060 | +0.001195 | [-0.001516, +0.003871] | tied |
| OOD-1000 | 999 | -0.002417 | -0.002034 | [-0.005912, +0.002007] | average improves; Gap tied |
| P8 targeted hard | 981 | +0.002582 | +0.004483 | [+0.000775, +0.008032] | significant Gap regression |
| PCQM4Mv2 valid 5k | 5,000 | n/a | +0.004238 | [+0.001093, +0.007424] | significant Gap regression |

Absolute common-set MAE is `0.102699 -> 0.102759 eV` (average) and
`0.119354 -> 0.120549 eV` (Gap) from v1 to v2. The OOD-average gain does not
offset statistically supported regressions on the targeted hard block and
PCQM-like transfer set.

## Artifacts

- `repair_v2_2d_common_metrics.json`
- `repair_v2_2d_common_predictions.csv`
- `repair_v2_2d_pcqm_metrics.json`
- `repair_v2_2d_pcqm_predictions.csv`
- `molgap-repair-v2-pure-2d-control-eval.log`

The SCNet encoder/head checkpoints and their per-run metrics remain under
`results/phase8/repair_v2_scnet/controlled_2d/`. The Kaggle evaluator is
`scripts/phase8/archive/remote/kaggle/molgap_repair_v2_2d_control_eval/`.
