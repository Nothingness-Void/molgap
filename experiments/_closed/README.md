# Phase 8 Negative Experiment Archive

These experiments are closed and are not candidates for the production model.
Their code and complete results remain available for reproducibility. Do not
rerun them unless the experiment is explicitly reopened with a new hypothesis.

| Experiment | Decision | Evidence |
|---|---|---|
| archive-r01 learned Router | Failed external promotion gates; keep fixed routed-v4 | [`archive-r01-learned-router/decision.md`](archive-r01-learned-router/decision.md) |
| archive-r02 PubChemQC Router | Oracle headroom exists, but pre-Expert gain is not learnable; sealed sets remain unopened | [`archive-r02-pubchemqc-router/decision.md`](archive-r02-pubchemqc-router/decision.md) |
| archive-r02 Late Soft Blend | Gap gain was 0.000881 eV, below the 0.001 eV gate | [`archive-r02-pubchemqc-router/late_blend_decision.md`](archive-r02-pubchemqc-router/late_blend_decision.md) |
| archive-r03 three-expert MoE | SchNet was weak and Router gain was below threshold | [`archive-r03-three-expert-moe/pilot_decision.md`](archive-r03-three-expert-moe/pilot_decision.md) |
| archive-r04 static dual-2D blend | Internal complementarity did not transfer across every frozen seed/block; do not scale | [`archive-r04-static-dual2d/external_transfer_decision.md`](archive-r04-static-dual2d/external_transfer_decision.md) |
| archive-r05 physics-consistent head | Exact label algebra improves output consistency but regresses external Gap MAE | [`archive-r05-physics-consistency/p0_physics_consistency_decision.md`](archive-r05-physics-consistency/p0_physics_consistency_decision.md) |
| archive-r06 structural GPS adaptor | G0 finds no interpretable ring/conjugation enrichment in the routed-v4 worst Gap decile | [`archive-r06-structural-gps-adapter/decision.md`](archive-r06-structural-gps-adapter/decision.md) |
| archive-r07 exact-2M encoder transplant | Preserving the 500K routed-v4 architecture still regressed all three paired seeds | [`archive-r07-exact2m-encoder-transplant/decision.md`](archive-r07-exact2m-encoder-transplant/decision.md) |
| archive-r08 full-1M routed fusion | The full-1M always-dual fusion reproduced, but the fixed 4 eV route regressed average and Gap MAE | [`archive-r08-full1m-routed-fusion/decision.md`](archive-r08-full1m-routed-fusion/decision.md) |
| archive-r09 original-1M late Router | Fixed alpha was a numerical tie and learned alpha regressed; original test remained locked | [`archive-r09-original1m-late-router/decision.md`](archive-r09-original1m-late-router/decision.md) |

Earlier Phase 8 work predating the numbered archive uses a separate namespace
and does not consume archive experiment numbers:

| Legacy family | Decision | Location |
|---|---|---|
| 30K controlled pilots | Coverage signal was weak-positive; MoE and tail routing were not promoted | [`legacy/pilots_30k/`](legacy/pilots_30k/) |
| Head and post-hoc probes | Layer fusion, residual calibration, weighted fusion, and head swaps failed promotion gates | [`legacy/head_posthoc/`](legacy/head_posthoc/) |
| Conformer ensemble | Accuracy improved, but roughly 6.8x inference cost keeps it opt-in | [`legacy/conformer_ensemble/`](legacy/conformer_ensemble/) |

The production recommendation remains the routed dual-GPS v4 recorded in
`CURRENT_STATE.md`.
