# Phase 8 1M Continuation

This directory contains lightweight reports for the 1M continuation. Large
graphs and frozen embeddings are in `data/cache/phase8/expansion_1m/`; candidate
checkpoints are in `models/phase8/expansion_1m/`.

The candidate is not a production model until it has a common-eval decision and
an explicit registry entry. Current production truth remains `CURRENT_STATE.md`.

## Handoff inventory

- `input_build/`: source-pool logs and report for the appended 500K molecules.
- `validation_schnet_local.json`: preserved local validation report from before
  import; its paths intentionally point at the original download location.
- The matching 1M GPS7/GPS9 weights are in the external-eval Kaggle model bundle.
  The remaining gate is full shared external evaluation, not weight recovery.
- `common_eval_kaggle_decision.md`: completed shared OOD-1000 and P8-hard
  acceptance decision. The candidate improves P8-hard strongly but does not
  replace routed-v4 globally because broad OOD does not improve.
- `pcqm4mv2_valid_5k_descriptor_analysis.md`: structural stratification of the
  public-valid regression; the main loss is small/low-aromatic/flexible rows.
- `pcqm4mv2_valid_5k_component_decision.md`: component-level diagnosis showing
  that the externally significant PCQM regression is concentrated in the 1M
  dual-GPS fusion calibration, not a uniform encoder collapse.
- `replay_fusion_decision.md`: frozen-embedding replay-weighted FusionHead
  follow-up. It is rejected because its PCQM public-valid regression remains
  `+0.012819 eV` relative to routed-v4.
