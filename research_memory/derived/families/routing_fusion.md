# routing_fusion (router, expert routing, fusion)

## Quick answer
Recent closed (of 2):
TB-pcqm-pairnorm-prediction-router-500k-20261027 | NO_TRAIN | pcqm-pairnorm-prediction-router-500k | existing-hard-and-convex-soft-cross | null | below_gate
TB-pcqm-expert-oracle-feasibility-20260930 | NO_TRAIN | pcqm-prediction-disagreement-expert-routing | saved-prediction-only-postdispatch-hard-soft | null | -
Open (ACTIVE) 0: -
Reopen: new_accepted_pretraining_artifacts_and_independently_qualified_role / New input features or independent role with separately prospective authorization
src (src/molgap/): conservative_fusion_payload.py, conservative_fusion_runner.py, fusion.py, hierarchical_fusion.py, multi2d.py, multi2d_router_fusion.py, pcqm_expert.py, pcqm_k1_gptrans_fusion.py, pcqm_route_b_fusion.py, qm9_fusion.py, +3 more
Experiments: multi2d_experts, pcqm_expert_oracle_feasibility, pcqm_full_minus_tail_fusion_20261113, pcqm_gine_expert, pcqm_k1_consistency_fusion_transfer, pcqm_k1_fusion_distillation, pcqm_k1_gptrans_full_fusion, pcqm_k1_gptrans_pairnorm_fusion_500k, pcqm_k1_hop_fusion_qualification, pcqm_pairnorm_prediction_router_500k

## Closed (2)
TB-pcqm-pairnorm-prediction-router-500k-20261027 | NO_TRAIN | pcqm-pairnorm-prediction-router-500k | existing-hard-and-convex-soft-cross | null | below_gate
TB-pcqm-expert-oracle-feasibility-20260930 | NO_TRAIN | pcqm-prediction-disagreement-expert-routing | saved-prediction-only-postdispatch-hard-soft | null | -

## Open (0)

## Levers (first 10)
family | arm | outcome | effect_meV | failure_mode | mechanism
pcqm-pairnorm-prediction-router-500k | 20261027 | NO_TRAIN | null | below_gate | existing-hard-and-convex-soft-cross
pcqm-prediction-disagreement-expert-routing | pcqm-expert-oracle-feasibility-20260930 | NO_TRAIN | null | - | saved-prediction-only-postdispatch-hard-soft

## Reopen conditions
new_accepted_pretraining_artifacts_and_independently_qualified_role
New input features or independent role with separately prospective authorization

## src
src/molgap/conservative_fusion_payload.py
src/molgap/conservative_fusion_runner.py
src/molgap/fusion.py
src/molgap/hierarchical_fusion.py
src/molgap/multi2d.py
src/molgap/multi2d_router_fusion.py
src/molgap/pcqm_expert.py
src/molgap/pcqm_k1_gptrans_fusion.py
src/molgap/pcqm_route_b_fusion.py
src/molgap/qm9_fusion.py
src/molgap/route_b_fusion.py
src/molgap/router.py
src/molgap/router_sampling.py

## Experiments
- experiments/multi2d_experts: scripts -
- experiments/pcqm_expert_oracle_feasibility: scripts analyze.py, finalize_diagnostic.py, prepare.py
- experiments/pcqm_full_minus_tail_fusion_20261113: scripts -
- experiments/pcqm_gine_expert: scripts accept_pcqm_expert.py, continue_pcqm_gine_local.py, scale_pcqm_gine_local.py
- experiments/pcqm_k1_consistency_fusion_transfer: scripts close.py, finalize_export.py, prepare.py, run.py
- experiments/pcqm_k1_fusion_distillation: scripts -
  - trace experiments/pcqm_k1_fusion_distillation/kaggle3_v1/distill_strong/rml_finalized/trace.json
  - trace experiments/pcqm_k1_fusion_distillation/kaggle3_v1/distill_weak/rml_finalized/trace.json
- experiments/pcqm_k1_gptrans_full_fusion: scripts accept_gptrans_full.py, run_fusion_study.py, run_gptrans_full.py
- experiments/pcqm_k1_gptrans_pairnorm_fusion_500k: scripts analyze.py, close.py, prepare.py
- experiments/pcqm_k1_hop_fusion_qualification: scripts run.py
- experiments/pcqm_pairnorm_prediction_router_500k: scripts analyze.py, close.py, prepare.py
