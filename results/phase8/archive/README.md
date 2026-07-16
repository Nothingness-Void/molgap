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

The production recommendation remains the routed dual-GPS v4 recorded in
`CURRENT_STATE.md`.
