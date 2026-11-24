# residual_corrector (residual corrector, residual block)

## Quick answer
Recent closed (of 3):
TB-pcqm-residual-spread-fit | NEGATIVE_UNDER_CONTRACT | pcqm-residual-spread-fit | no-model-change-fit-p0-s | -0.73 | below_gate
TB-pcqm-residual-block-transfer | NEGATIVE_UNDER_CONTRACT | pcqm-residual-block-transfer | no-model-change-keep-p0-s | +2.27 | endpoint_regression
TB-pcqm-pairnorm-residual-2d-p0 | NO_TRAIN | pcqm-pairnorm-residual-2d-p0 | single-fixed-2d-chemistry-residual-corrector | null | -
Open (ACTIVE) 0: -
Reopen: New prospective independent holdout design and explicit authority; no automatic H1 or protected-role / New prospective record and explicit authority for fit rows spread over the index range; no automatic / Closed for post-hoc 2D descriptor residual trees on these backbones; only a genuinely different mech
src (src/molgap/): residual_attribution.py
Experiments: pcqm_k1_consistency_residual_screen, pcqm_k1_initial_residual_100k, pcqm_k1_residual_reconciliation, pcqm_pairnorm_residual_2d_p0, pcqm_residual_block_transfer, pcqm_residual_spread_fit

## Closed (3)
TB-pcqm-residual-spread-fit | NEGATIVE_UNDER_CONTRACT | pcqm-residual-spread-fit | no-model-change-fit-p0-s | -0.73 | below_gate
TB-pcqm-residual-block-transfer | NEGATIVE_UNDER_CONTRACT | pcqm-residual-block-transfer | no-model-change-keep-p0-s | +2.27 | endpoint_regression
TB-pcqm-pairnorm-residual-2d-p0 | NO_TRAIN | pcqm-pairnorm-residual-2d-p0 | single-fixed-2d-chemistry-residual-corrector | null | -

## Open (0)

## Levers (first 10)
family | arm | outcome | effect_meV | failure_mode | mechanism
pcqm-pairnorm-residual-2d-p0 | pcqm-pairnorm-residual-2d-p0 | NO_TRAIN | null | - | single-fixed-2d-chemistry-residual-corrector
pcqm-residual-block-transfer | pcqm-residual-block-transfer | NEGATIVE_UNDER_CONTRACT | +2.27 | endpoint_regression | no-model-change-keep-p0-s
pcqm-residual-spread-fit | pcqm-residual-spread-fit | NEGATIVE_UNDER_CONTRACT | -0.73 | below_gate | no-model-change-fit-p0-s

## Reopen conditions
New prospective independent holdout design and explicit authority; no automatic H1 or protected-role access.
New prospective record and explicit authority for fit rows spread over the index range; no automatic launch.
Closed for post-hoc 2D descriptor residual trees on these backbones; only a genuinely different mechanism under a new protocol reopens residual work.

## src
src/molgap/residual_attribution.py

## Experiments
- experiments/pcqm_k1_consistency_residual_screen: scripts finalize_rml.py, prepare_rml.py, run_analysis.py
- experiments/pcqm_k1_initial_residual_100k: scripts -
  - trace experiments/pcqm_k1_initial_residual_100k/runs/pcqm-k1-initial-residual-100k-s42/prospective/baseline/rml_finalized/trace.json
  - trace experiments/pcqm_k1_initial_residual_100k/runs/pcqm-k1-initial-residual-100k-s42/prospective/initial_residual/rml_finalized/trace.json
- experiments/pcqm_k1_residual_reconciliation: scripts analyze.py
- experiments/pcqm_pairnorm_residual_2d_p0: scripts chemistry.py, probe.py
- experiments/pcqm_residual_block_transfer: scripts transfer.py
- experiments/pcqm_residual_spread_fit: scripts spread.py
