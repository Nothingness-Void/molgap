# Release decision — 2026-09-14

The accepted [relation-flow round](../pcqm_gptrans_relation_flow/decision.md)
shortlisted pair pre-normalization and rejected logit centering. This justified
one distinct information-flow question, not stacking a marginal winner with a
failed arm. [Release facts](results/release.json) bound the remaining authority.

Code inspection found that the GPTrans pair residual stores `P + delta`, while
the direct pair-to-node message pools only `delta`. History still influences
attention logits; it is not absent from the model. The unanswered question is
whether a direct value/readback path from accumulated relations helps beyond
that indirect path. This is neither a new slot nor a repeated shortest-path,
geometry, directed-bond or K1-relation-slot mechanism.

Two isolated variants were selected to distinguish payload and routing:
`memory_value` uses original delta-derived weights on accumulated pair values;
`memory_message` uses accumulated pairs for both weights and values. Both use
the untouched reference's raw pair/logit pathways, not PreNorm or centering.
They retain all seed-42 tensors and parameters. Full V4 optimization is frozen.

If both regress, direct persistent-pair readback closes; no automatic scaling,
gate, normalization, seed or third-round rescue is released. A material gain
nominates a mechanism only. EMA-selected last-epoch behavior and repeated-dev
selection limit causal and generalization claims. The PreNorm result remains
independently shortlisted regardless of this result.
