# Cardinality-channel evidence selection

The post-EdgeState evidence review selected one untested, pure-2D information
channel from the 2026 literature audit.

## Why this question

- Full GPS attention is useful at scale but repeatedly spends capacity on
  normalized global averages; GraphState evidence shows that more dense global
  mixing is not automatically better.
- Uniform 2/3-hop aggregation was weakly positive, while learned softmax
  relative-value attention regressed. Both normalize or select among path
  values and can erase how many structurally valid sources support an atom.
- CardinalGraphFormer isolates a different mechanism: a query-gated,
  unnormalized value sum over the exact same local support preserves support
  cardinality. Its explicit-size ablation motivates a causal control rather
  than an unsupported module transplant.

The selected screen therefore retains the accepted pure-2D EdgeState GPS9 and
tests one shared K<=3 cardinality channel after layers 3/6/9. A parameter-
matched target-state-times-size control distinguishes neighbor content from a
simple molecule/neighborhood-size shortcut.

## Exclusions

This is not the closed sparse relative-value path attention, PairGPS, ring or
conjugated ComponentState, pretraining, model widening, ETKDG geometry,
teacher distillation, residual prediction, or fusion. No desktop 304-wide or
500K+ run is used as a comparator.

