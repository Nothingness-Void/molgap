# Evidence-led follow-up review — 2026-10-10 JST

This is literature/design evidence, not an executed experiment, compute release,
terminal trajectory or Replay admission. The user explicitly required weakness
localization and keyword-driven primary-paper research before another submission.
The completed portability result and its frozen gate remain unchanged.

## Observed weakness versus unmeasured mechanism

The [portability decision](decision_terminal.md) supports a cohort/sampling
component: relative gain changed without optimizer updates after original
checkpoint predictions reproduced. It does not identify size, degree, scaffold,
source concentration or gate amplitude as the cause. The reused panel is not a
fresh independent holdout.

The [triplet trajectories](../pcqm_gptrans_triplet_communication_100k/gpu/results/interpretation.md)
support connected, sustained relation gains but late development erosion while
training fit improved. Query-dependent attention did not establish superiority
over aggregation. This motivates a generalization investigation, not another
disconnection repair or an assumed exposure deficit.

Related evidence rules out easy explanations:

- [Node width / FFN expansion](../pcqm_gptrans_capacity_nodes_100k/gpu/results/interpretation.md)
  did not improve the endpoint at equal exposure.
- [Local amplitude control](../pcqm_gptrans_local_control_100k/gpu/results/interpretation.md)
  worked as specified but worsened development; magnitude alone does not justify
  another suppression mechanism.
- [Reduced decay](../pcqm_gptrans_decay_clock_100k/gpu/results/interpretation.md)
  did not repair generalization; do not reopen a coefficient grid.
- The [local-stream 500K result](../pcqm_gptrans_local_transfer_500k/gpu/results/interpretation.md)
  retained benefit. Scaling is not universally destructive; that endpoint was
  about twelve passes, not complete convergence.

No accepted diagnostic measures source concentration or triplet gate mass on
both cohorts. Neither is a demonstrated bottleneck.

## Primary papers, methods and ablations checked

Queries targeted triplet gated aggregation, graph source dropout, dropout before
softmax, molecular generalization and graph scaling. Sources are indexed in
[the machine-readable review](results/literature_followup.json).

| Source | Concrete finding | Transfer limitation |
|---|---|---|
| [TGT, ICML 2024](https://arxiv.org/html/2402.04538v2), section 3.1, tables 9/10, appendix D.1, official code | Sigmoid after softmax is intentional; removing gates slightly worsened distance prediction. Source masking has a separate ablation. | Geometry training, EGT backbone and published scores are not the local pure-2D recipe. |
| [DropKey, CVPR 2023](https://arxiv.org/html/2208.02646v4), equations 13/14 and generalization/schedule/expectation ablations | Pre-softmax source exclusion differs from post-softmax dropout. | Vision evidence is motivation only; its schedule and extra finetuning cannot be silently imported. |
| [Graph scaling, LoG/PMLR 2025](https://arxiv.org/html/2402.02054v3), sections 5/6, appendix I | Equal graph counts can hide different structural data volumes; node/edge counts add information. | Extra exposure telemetry is appropriate, but does not replace frozen optimizer steps or presentations or supply a dense-GPTrans compute law. |
| [Molecular scalability, NeurIPS 2024](https://arxiv.org/html/2404.11568v3), sections 5.2/5.3, appendices E.2/E.3 | Task alignment, label diversity and probing choices affect scaling. | Multitask foundation-model findings are not a direct-Gap fix or a reason to reopen width/depth grids. |

[Official TGT attention](https://github.com/shamim-hussain/tgt/blob/master/lib/tgt/layers/layers.py)
samples one source mask shared across queries/heads per graph/layer before
normalization. It does not remove input atoms or actual chemical bonds.
[Official triplet code](https://github.com/shamim-hussain/tgt/blob/master/lib/tgt/layers/triplet.py)
preserves post-softmax gates. Its triplet dropout is a distinct operation.
GPTrans has different node/pair paths; a faithful adaptation needs an explicit
scope check, not copying an isolated mask or a benchmark score.

## Candidate disposition

1. **Gate removal / renormalization not released.** The algebra is not an
   implementation defect and no measured gate-mass shift links it to attenuation.
   This rejects the proposed rationale, not every possible normalized-gate model.
2. **Source regularization retained as the conditional priority.** It can change
   source reliance without adding inference parameters. The local late erosion
   plus paper ablation justify investigation, not a proven fix.
3. **No width, FFN, cap, extra-seed or geometry variant released.** New evidence
   needed to reopen those questions is absent.

## Cheapest decision-relevant assay to prepare

Freeze a separate NO_TRAIN source-dependence assay over the accepted local parent
and aggregation checkpoint. Reuse native frozen loading, accepted graph mounts,
reproduction, atomic chunks, unchanged-state checks, cost accounting and RML
diagnostic closure. No local inference, fitting, optimizer, checkpoint reselection
or protected role. This assay is proposed, not prepared or submitted.

- Select source-index panels on original and later fixed internal development
  using only input identity; never select by targets, residuals or winners.
- Observe GPA entropy, maximum source mass, virtual-source mass and effective
  source fraction by layer/head. Separate real-source-conditional statistics:
  useful virtual-token attention is not automatically collapse.
- Record input graph size/degree distributions before claiming structural shift.
  Observe inward/outward triplet softmax mass, gated mass and return/input RMS
  without changing masks, normalization, weights, buffers or RNG. Exclude padding.
- Concentration alone is not a defect. Cohort-conditioned observations can
  weaken or retain the hypothesis, not prove overfitting or train a Router.
  Frozen destructive ablation is not the net value of a retrained module.
- No assay result automatically releases successor training.

If justified afterwards, the minimal training question is source-dropout
replacement on unchanged aggregation against its exact immutable aggregation
reference, not a weaker scalar baseline. Freeze probability, mask scope, virtual
token policy, all affected GPA paths, RNG/resume semantics, parameter identity
and evaluation equivalence. Preserve data, FP32, BS128, seed42, optimizer, loss,
EMA, schedule, selection and exposure; no dropout grid, finetuning stage or
simultaneous gate change. The applicable comparator/release gate must pass.

A future 100K screen also prospectively packages its separate full canonical
50K later-role audit under [screening policy](../SCREENING_POLICY.md). Training
and audit close separately. Diagnostic evidence remains NO_TRAIN; planning does
not grant terminal Replay readiness.

## Execution boundary

No remote job, upload or monitor binding was created during this review. The
prior terminal monitor remains paused. No model, training contract, production
registry, desktop workflow or historical scientific decision was modified.
