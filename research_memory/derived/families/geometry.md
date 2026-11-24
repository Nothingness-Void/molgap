# geometry (3D geometry, distance angle, DFT distance)

## Quick answer
Last closed (of 4, 0 are CLOSED reference arms):
TB-pcqm-geometry-reliability-screen-20260928 | POSITIVE_UNDER_CONTRACT | pcqm-geometry-reliability-screen | no-model-change-one-global-convex | null | -
TB-pcqm-geometry-component-attribution-20260928 | NO_TRAIN | pcqm-geometry-component-attribution | no-model-change-hash-gated-analysis | null | -
TB-geometry-transfer-500k-context | INCONCLUSIVE | distance-angle-transfer | distance-angle-geometry-features | null | -
Open (ACTIVE) 0: -
Reopen: separate V5 transfer qualification and budget decision / explicitly authorized matched V4 or V5 bridge / A separately frozen strict component contract with a qualified 2D comparator, accepted executable ca
src (src/molgap/): geometry_features.py, pcqm_distance_angle_scale.py, pcqm_geometry_transfer.py, pcqm_geometry_transfer_runner.py
Experiments: pcqm_distance_angle_500k, pcqm_geometry_component_attribution, pcqm_geometry_reliability_gate, pcqm_geometry_transfer_500k, pcqm_gptrans_geometry_channels_100k_kaggle1

## Closed (4)
TB-pcqm-geometry-reliability-screen-20260928 | POSITIVE_UNDER_CONTRACT | pcqm-geometry-reliability-screen | no-model-change-one-global-convex | null | -
TB-pcqm-geometry-component-attribution-20260928 | NO_TRAIN | pcqm-geometry-component-attribution | no-model-change-hash-gated-analysis | null | -
TB-geometry-transfer-500k-context | INCONCLUSIVE | distance-angle-transfer | distance-angle-geometry-features | null | -
TB-distance-angle-triangle-500k-20260911 | POSITIVE_UNDER_CONTRACT | distance-angle-triangle-500k | candidate-adds-etkdg-distance-and-angle | null | -

## Open (0)

## Levers (first 10)
family | arm | outcome | effect_meV | failure_mode | mechanism
distance-angle-transfer | geometry-transfer-500k-context | INCONCLUSIVE | null | - | distance-angle-geometry-features
distance-angle-triangle-500k | 20260911 | POSITIVE_UNDER_CONTRACT | null | - | candidate-adds-etkdg-distance-and-angle
pcqm-geometry-component-attribution | 20260928 | NO_TRAIN | null | - | no-model-change-hash-gated-analysis
pcqm-geometry-reliability-screen | 20260928 | POSITIVE_UNDER_CONTRACT | null | - | no-model-change-one-global-convex

## Reopen conditions
separate V5 transfer qualification and budget decision
explicitly authorized matched V4 or V5 bridge
A separately frozen strict component contract with a qualified 2D comparator, accepted executable cache, budget, preflight, and replay-complete per-arm evidence.
A separately frozen geometry-capable matched training contract and strict reference qualification.

## src
src/molgap/geometry_features.py
src/molgap/pcqm_distance_angle_scale.py
src/molgap/pcqm_geometry_transfer.py
src/molgap/pcqm_geometry_transfer_runner.py

## Experiments
- experiments/pcqm_distance_angle_500k: scripts accept_terminal.py
- experiments/pcqm_geometry_component_attribution: scripts analyze.py
- experiments/pcqm_geometry_reliability_gate: scripts prepare_terminal.py, screen.py
- experiments/pcqm_geometry_transfer_500k: scripts -
- experiments/pcqm_gptrans_geometry_channels_100k_kaggle1: scripts -
  - trace experiments/pcqm_gptrans_geometry_channels_100k_kaggle1/prospective_attempt_002/distance_angle/rml_finalized/trace.json
  - trace experiments/pcqm_gptrans_geometry_channels_100k_kaggle1/prospective_attempt_002/distance_only/rml_finalized/trace.json
