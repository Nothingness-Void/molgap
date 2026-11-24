# scale_transfer (scale transfer, 100K to 500K, total budget)

## Quick answer
Recent closed (of 6):
TB-pcqm-scale-fit-retained-20261001 | NO_TRAIN | pcqm-scale-fit-retained | no-model-or-optimizer-change-six | null | -
TB-pcqm-scale-transfer-reassessment-20260925 | NO_TRAIN | pcqm-scale-transfer-postmortem | no-model-change-backtest-earlier-rml | null | -
TB-pcqm-scale-transfer-diagnostic | POSITIVE_UNDER_CONTRACT | pcqm-scale-transfer-diagnostic | evidence-based-quantitative-cross-scale-exposure | null | -
Open (ACTIVE) 0: -
Reopen: A new matched, budget-bounded causal comparison is frozen and separately authorized. / new_decision_relevant_matched_factorial_contract
src (src/molgap/): pcqm_500k_v4_ablation.py, pcqm_500k_v4_evidence.py, pcqm_distance_angle_scale.py, pcqm_k1_scale.py, pcqm_k1_scale_runner.py, scale_transfer_diagnostic.py, substrate_scale_addon.py
Experiments: pcqm_500k_v4_evidence, pcqm_k1_scale_attribution_20261025, pcqm_scale_fit_retained, pcqm_scale_transfer_attribution, pcqm_scale_transfer_diagnostic, pcqm_scale_transfer_reassessment, pcqm_single_model_total_budget

## Closed (6)
TB-pcqm-scale-fit-retained-20261001 | NO_TRAIN | pcqm-scale-fit-retained | no-model-or-optimizer-change-six | null | -
TB-pcqm-scale-transfer-reassessment-20260925 | NO_TRAIN | pcqm-scale-transfer-postmortem | no-model-change-backtest-earlier-rml | null | -
TB-pcqm-scale-transfer-diagnostic | POSITIVE_UNDER_CONTRACT | pcqm-scale-transfer-diagnostic | evidence-based-quantitative-cross-scale-exposure | null | -
TB-pcqm-scale-transfer-attribution | NO_TRAIN | pcqm-scale-transfer-attribution | no-model-change-evidence-only-decomposition | null | -
TB-matched-500k-v4-three-arm | POSITIVE_UNDER_CONTRACT | matched-500k-architecture-comparison | matched-three-arm-architecture-comparison | null | -
T-pcqm-single-model-total-budget-500k-s42-b_16x256 | NEGATIVE_UNDER_CONTRACT | pcqm-single-model-total-budget-500k | architecture-package-system-performance | null (see experiments/pcqm_single_model_total_budget/terminal_decision.md) | -

## Open (0)

## Levers (first 10)
family | arm | outcome | effect_meV | failure_mode | mechanism
matched-500k-architecture-comparison | matched-500k-v4-three-arm | POSITIVE_UNDER_CONTRACT | null | - | matched-three-arm-architecture-comparison
pcqm-scale-fit-retained | 20261001 | NO_TRAIN | null | - | no-model-or-optimizer-change-six
pcqm-scale-transfer-attribution | pcqm-scale-transfer-attribution | NO_TRAIN | null | - | no-model-change-evidence-only-decomposition
pcqm-scale-transfer-diagnostic | pcqm-scale-transfer-diagnostic | POSITIVE_UNDER_CONTRACT | null | - | evidence-based-quantitative-cross-scale-exposure
pcqm-scale-transfer-postmortem | pcqm-scale-transfer-reassessment-20260925 | NO_TRAIN | null | - | no-model-change-backtest-earlier-rml
pcqm-single-model-total-budget-500k | s42-b_16x256 | NEGATIVE_UNDER_CONTRACT | null (see experiments/pcqm_single_model_total_budget/terminal_decision.md) | - | architecture-package-system-performance

## Reopen conditions
A new matched, budget-bounded causal comparison is frozen and separately authorized.
new_decision_relevant_matched_factorial_contract

## src
src/molgap/pcqm_500k_v4_ablation.py
src/molgap/pcqm_500k_v4_evidence.py
src/molgap/pcqm_distance_angle_scale.py
src/molgap/pcqm_k1_scale.py
src/molgap/pcqm_k1_scale_runner.py
src/molgap/scale_transfer_diagnostic.py
src/molgap/substrate_scale_addon.py

## Experiments
- experiments/pcqm_500k_v4_evidence: scripts accept_stage.py, analyze_local_ablation.py, package.py, package_final_full_gps.py, prepare_local_data.py, run_local_ablation.py
- experiments/pcqm_k1_scale_attribution_20261025: scripts close_failed_attempt.py, prepare.py, run.py
- experiments/pcqm_scale_fit_retained: scripts finalize_diagnostic.py, run_diagnostic.py
- experiments/pcqm_scale_transfer_attribution: scripts -
- experiments/pcqm_scale_transfer_diagnostic: scripts analysis.py
- experiments/pcqm_scale_transfer_reassessment: scripts analyze.py
  - trace experiments/pcqm_scale_transfer_reassessment/sources/edgestate/trace.json
  - trace experiments/pcqm_scale_transfer_reassessment/sources/k1/trace.json
- experiments/pcqm_single_model_total_budget: scripts prepare_inputs.py, profile_local.py
  - trace experiments/pcqm_single_model_total_budget/runs/pcqm-single-model-total-budget-500k-s42/prospective/a_12x384/rml_finalized/trace.json
  - trace experiments/pcqm_single_model_total_budget/runs/pcqm-single-model-total-budget-500k-s42/prospective/b_16x256/rml_finalized/trace.json
