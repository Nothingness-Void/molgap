# Relation-flow and alternative-family research — 2026-10-05

## Scope and disposition

This local review shortlisted one GPTrans intervention, one independent family,
and one lower-priority intervention. It did not release compute, implement a
model, run inference, query remote jobs, or take custody of the user-reported
Kaggle1 dual500K run. The identities of that run's arms were not locally bound.

Repository baseline: `f426350ad636d32ea05c059664cea35fa44f1618`.
The retained derived index contained82 trajectories; the Replay pool contained
41 entries in7 comparable groups. Entries include references and must not be
reported as41 independent complete experimental loops. This review inspected
the index and selected owning decisions, not every raw artifact or all papers.
The source digest was
`baebcf835afbe986bc7de975fa63ceabe51033b72f05a5b36f477cf4a00e2985`.

The proposals below are untested hypotheses. They are neither architecture
promotions nor new RML terminal trajectories. Source reading and algebraic
parameter counts are not runtime qualification.

## 1. Evidence that changed prioritization

| Retained experiment | Observation under its own contract | Consequence for research |
|---|---|---|
| [Node352 / FFN2](../pcqm_gptrans_capacity_nodes_100k/gpu/results/interpretation.md) | Versus G1 EMA999, candidate-minus-reference MAEs were+0.0028690870/+0.0002831951 eV. Steps and exposure matched. | Do not start another node-width or node-FFN grid. FFN2 was not a statistically established regression. |
| [True-bond local stream](../pcqm_gptrans_capacity_relations_100k/gpu/results/interpretation.md) | Gain0.0018744017 eV with11.9% more parameters; late train fitting improved while development gain narrowed. | Relation-specific flow remains plausible, but more fitting capacity is not sufficient and transfer is unproven. |
| [EMA correction](../pcqm_gptrans_input_ema_100k/gpu/results/decision.md) | Live optimization matched; fast EMA gained0.0066522625 eV. | Keep the accepted corrected selection reference; do not mistake filter lag for missing architecture capacity. |
| [Shared-live500K EMA](../pcqm_gptrans_capacity_relations_100k/gpu/results/interpretation.md) | Fast EMA gained0.0095257311 eV over slow EMA after46,860 updates. | This supports filter correction, not a500K local-addon result or a causal architecture Replay pair. |
| [Pair depth scaling](../pcqm_gptrans_pair_scale_100k/gpu/results/decision.md) | Uniform1/sqrt(12) scaling lost0.0025649499 eV; no baseline pair-activation trace existed. | Do not prescribe another uniform damping sweep or assert baseline pair explosion. |
| [K1 relation portability](../pcqm_k1_relation_resolution_100k/audit/decision.md) | Receiver, triplet and RRWP gains reversed on later molecules without changing weights. | An early/single-role gain is insufficient; information richness alone did not establish portability. |
| [Old sparse relative values](../pcqm_gap_architecture/results/relative_value_graphstate_seed42/decision.md) | GraphState path-conditioned values lost0.0013053970 eV and slowed throughput; interval crossed zero. | Relative values have an explicit local counterexample. A different dynamic-pair design needs a new justification, not a renamed retry. |
| [MetaGIN](../pcqm_metagin_2d_100k/attempt_v2/decision.md), [motif](../pcqm_motif_hierarchy_100k/attempt_v3/decision.md), [PNA statistics](../pcqm_gap_architecture/results/pna_statistics_graphstate_seed42/decision.md) | No supported promotion of the exact adaptations. | Do not reopen these implementations. Their losses do not falsify entire published families. |

No absolute MAE was compared across different development roles or recipes.
The immutable labels and experiment-specific material gates were not changed.

## 2. Static GPTrans finding: a narrow per-block relation update

The [local GPA owner](../../src/molgap/gptrans.py) and
[official GPA implementation](https://raw.githubusercontent.com/czczup/GPTrans/main/models/gptrans.py)
both map32 pair channels to8 attention heads, then expand the8 head signals
back to32 pair channels. The pair contribution to nodes also exists already:
it uses its own channel-wise softmax on the newly expanded pair update.
Thus it is false to describe this GPTrans as having no pair-to-node flow.

At one block, the pair-update vectors occupy the affine image of an8-to32
projection. This follows from the declared operator shapes, without loading
weights. It is **not** proof of representation collapse or a measured error
cause: the initial32-channel state is retained, and different block projections
can span different directions. The complete recurrent state need not have
rank8. Nevertheless, a full-channel nonlinear relation transition is a clean
untested alternative to widening the node stream.

## 3. Shortlist

### A. GPTrans pair-transition addon — first bounded falsifier

Hypothesis: pair memory may benefit from a nonlinear channel transition that
does not pass exclusively through the head-score expansion.

Proposed change: immediately **before GPA blocks3/6/9/12**, add an independent
32-to64-to32 GELU residual MLP, with LayerNorm only inside the new branch and
zero-initialized return projection. Apply it to valid real-atom pairs; preserve
virtual/padded positions and all existing GPA, node FFN and readout operators.
The last injection is before block12, not after it: otherwise a real-pair-only
branch could have no route to the final virtual-node readout.

Algebraic additional parameters per injection:
`2*32 + (32*64+64) + (64*32+32) = 4,256`.
Four injections add17,024 parameters, giving a proposed total5,263,841
(+0.3245% over5,246,817). These are design counts, not measured preflight counts.
No baseline/addon stacking or physical parameter expansion is implied.

This is not the previously stopped Pair PreNorm, uniform pair-depth scaling,
node FFN2, or an RRWP/path-input variant. Pair normalization in the original
GPA is not changed. The research inspiration is GRIT's nonlinear pair update,
not a claim to reproduce GRIT.

Keep G1 initialization, EMA0.999, all non-intervention scientific identities
and46,860-step100K endpoint. The immutable accepted reference is reused.
This candidate should be tested independently, not stacked with the local-bond
addon while its user-owned500K transfer experiment is unresolved.

Critical risks: all-pairs MLP work remains quadratic and may be slow despite
few parameters; extra nonlinear capacity may overfit; zero initialization does
not guarantee good later optimization. Require optimizer-inclusive preflight
and a prospectively approved cost ceiling, not a speed claim from parameter count.

Prospective diagnostics should include branch-return RMS/gradient norms, pair
update norms by depth, head entropy, train/live/EMA development, late matched-step
gains, and per-row errors. Parameter count alone is not the decision metric.

### B. Compact complete GRIT — independent-family reserve

[GRIT paper](https://proceedings.mlr.press/v202/ma23c/ma23c.pdf) and
[official layer](https://raw.githubusercontent.com/LiamMa/GRIT/main/grit/layer/grit_layer.py)
were rechecked beyond the abstract. The distinctive bundle is learned RRWP
pair initialization, nonlinear pair-conditioned attention with pair updates,
relation-aware values, and per-layer degree scalers with BatchNorm. The paper's
ZINC ablations separately removed these choices. Its PCQM result was a
single-run validation result, not proof of our budgeted performance.

Its published PCQM model used16 layers, width256,8 heads, RRWP16, mean pooling,
batch256 and150 epochs, about16.6M parameters. A proposed8-layer/192-channel
budget adaptation would preserve the family mechanisms but **not** reproduce
that model or its score. Exact parameter count and FP32/T4 runtime remain unknown.

This would replace the backbone, not add RRWP once at K1 layer6. The old RRWP
portability failure remains contrary evidence for cheap transplantation, but
does not constitute a whole-GRIT test. Pure2D RRWP must be deterministically
derived from the same accepted adjacency; no new split, coordinate source,
teacher, pretrained checkpoint or target-derived structural label is permitted.

Before release, inspect the complete official input/readout path, preserve OGB
categorical features, specify degree statistics from real bonds rather than the
completed attention graph, mask padding in normalization, and ensure evaluation
BatchNorm does not update or depend on development batch composition. EMA buffer
handling must be explicitly compatible with this BN architecture. These are
family integration questions, not reasons to silently alter the baseline recipe.

Reuse the [shared graph adapter owner](../../ARCHITECTURE.md#registered-family-lifecycle)
and the established release/acceptance/RML owners rather than importing the
entire GraphGym training stack. Verify actual module dependencies first. A
complete source/factory/output/runtime qualification is still absent.

### C. GPTrans shared-route relation values — lower priority

[GRPE's graph-encoded value equation and ZINC ablation](https://arxiv.org/html/2201.12787)
provide mechanism evidence, not a guaranteed PCQM gain. The proposed design
keeps the original GPA paths and adds `sum_j alpha_ij^h W_h p_ij` to the
existing node-value stream at blocks3/6/9/12. It uses the existing head routing
weights and the current recurrent32-channel pair state; it does not compute
new queries, path features, independent routing weights or geometry.

Bias-free32-to32 return matrices for8 heads add8,192 parameters per injection,
or32,768 total: proposed total5,279,585. Zero-initialize returns; contract and
padding semantics require review before implementation. Reduce weighted pair
states before projection to avoid materializing N-by-N-by256 values.

This differs from the old GraphState sparse static2/3-hop path-value addon.
However, GPTrans already has independent pair-to-node communication, and the
old value-path and K1 receiver losses lower confidence. Do not call this missing
information or run it automatically just because a paper used relative values.
Prefer A first; retain C only if A's diagnostics or a separate binding audit
supports a routing/content mismatch.

## 4. Leads withheld after source audit

- [ESA full text](https://arxiv.org/html/2402.10793v3) reports a400-epoch single
  PCQM run on public validation-as-test with no same-run baselines. That
  alone does not demonstrate leakage, but it cannot be advertised as an
  accepted OGB hidden-test result or a matched MolGap improvement. The paper
  also reports PNA ahead on QM9 frontier-orbital targets. Its mechanism was
  not promoted merely because of its headline score.
- [TetraGT official repository](https://github.com/xkxxfyf/TetraGT) still had
  only README/assets and stated code forthcoming at this review. Its
  advertised pipeline includes conformer prediction and geometry pretraining,
  with60M-plus models. A2D inference label alone does not establish a
  geometry-free training contract. No direct budgeted reproduction was selected.
- Higher-order cubic attention, another node-width/FFN sweep, motif/PNA-stat
  retries and path+EMA stacking were not released.

## 5. Evidence and scale guardrails before a later execution decision

No accepted single-role gain or paper score guarantees100K-to500K retention.
Separate: (1) portability with frozen weights to a predeclared later role;
(2) actual500K optimization at a matched exposure; and (3) eventual full-budget
learning. A cheap frozen-weight audit cannot replace(2) or(3).

For a later100K screen, bind the immutable G1 EMA999 reference, source/init and
fixed cross-platform data, FP32/no-TF32 BS128 seed42, optimizer, schedule,
transform, selection and endpoint. Freeze a separate material/cost rule from
the applicable V5 policy before results; do not declare0.003 universal or move
a previous experiment's threshold after inspecting its outcome.

Any later role check needs explicit membership, once/read/reuse history and
reference predictions. Do not create an unbound random10K slice or reuse a
development audit as sealed validation. Future acceptance must verify real
prospective->trace->artifact->role/cost->terminal bindings and **actual** Replay
candidate/reference admission; this research note cannot supply them.

The suggested research order was A, then B if a distinct-family budget was
approved; C remained conditional. No corresponding training was submitted.
