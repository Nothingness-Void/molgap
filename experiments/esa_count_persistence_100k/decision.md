# Research selection — 2026-10-25

The human requested historical RML analysis and a new bounded submission.
The coordinator selected one within-ESA persistence comparison, rather than
another PPGT conditioner, generic widening or a stack of marginal K1 addons.
This record freezes the selection rationale, not an experimental outcome.

## Source inventory and limits

The existing `molgap.research_curve_review` reader validated source metadata
and reconstructed eligible pairs in memory, without changing user-derived
indexes or running a model. The inventory contained165 trajectories,
130 V5 envelopes,64 trace manifests,17 reference bundles,52 replay entries
in10 comparison groups and42 candidate/reference curve contrasts. ACTIVE32
included blocked and unsubmitted plans; it was not a count of remote jobs.
Replay entries include references and are not52 completed model discoveries.

Full hash-bound review: `E:/MolGap/artifacts/route-review-20261025/curve_review.json`,
SHA256 `368c41059be7007b4b7626c5e58a567032f449eb15065c1c8513d9612cb1dd73`.
This is a working-tree source snapshot, not merely a Git HEAD. The retained
derived snapshot disagreed with source-built eligibility; it was not rebuilt.

## What changed the decision

| Evidence | Interpretation for the selected question |
|---|---|
| [K1 frozen-scale assay](../pcqm_k1_cross_scale_frozen/decision.md): original gain0.003044eV fell to0.000416 on later molecules before retraining; matched500K later reversed it | Exposure alone cannot explain all scale attenuation; avoid treating a tiny selected100K gain as transferable |
| [GPTrans500K local stream](../pcqm_gptrans_local_transfer_500k/gpu/results/interpretation.md): gain0.005650eV, complete matched equal-update evidence | Useful relationship flow exists; this is about12 passes, not proof of asymptotic/full-scale gain |
| [Earlier curve synthesis](../pcqm_v5_route_portfolio/rml_route_reserve_2026-10-23.md): triplet attention did not beat cheaper aggregation; width, feedback, route changes and positive-module combinations often failed | Do not repeat these closed mechanisms or invent another sweep |
| [Modern2D structural endpoints](../modern2d_structure_100k/results/decision.md): PPGT lead contracted0.007259 atpass10 to0.001318 atpass20; ESA count led18/20 points, selected gain0.004364 | Retain only ESA count as a bounded hypothesis; the old context-only comparison is not a causal reference |
| Same ESA record: target<4eV rows worsened0.009835 while other target bins improved; removing median bias did not erase the gain | Do not call count a general solution; inspect regressions and exclude same-role calibration as a deployable improvement |
| [ESA LR attribution](../modern2d_configured_100k/esa_attribution/lr/results/decision_2026_10_21.md): LR3e-4 was the strongest retained partial lead | Freeze3e-4 in both arms rather than confound a readout intervention with optimizer tuning |

The [ESA paper](https://pmc.ncbi.nlm.nih.gov/articles/PMC12141427/)
describes alternating masked/self attention over edges and learned-seed
attention pooling. Its pooling processes a fixed learned seed set rather than
an explicit graph-size statistic (Methods, equations8–10). The inference here
is narrow: supplying size after normalized pooling can expose information
missing from that readout. The paper does not establish our addon or a PCQM
gain. `CountConditionedReadout` implements an existing zero-start log-count
FiLM; it adds512 parameters and does not add geometry or another encoder.

## Why this is a new falsifier

The historical20-pass screen did not answer whether the count lead survives
60 passes or provide an accepted reusable reference. A cosine schedule with
60 passes is not a continuation of a20-pass cosine run. Both arms therefore
start from the same frozen state under a newly declared, matched60-pass
contract. This one baseline enrollment is the exception needed to establish
the immutable reference, not permission to retrain it for later candidates.

The anchor copies the existing ESA data/model declaration with LR3e-4 frozen
instead of LR1e-3. It is source provenance, not fabricated terminal evidence.
No old trajectory or negative decision was upgraded. No cross-family MAE
ranking,500K, full training, extra seed or protected-role release follows.
