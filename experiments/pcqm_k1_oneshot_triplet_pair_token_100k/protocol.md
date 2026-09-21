# Frozen protocol: K1 one-shot triplet PairToken

## Question

Can one bounded TGT-style directed bond-to-bond interaction improve the
accepted K1 PairToken without the overfit and cost of persistent full-depth
triplet state?

## Evidence basis

- Original PairToken is the only retained K1 relation mechanism that cleared
  the fixed 100K `0.003 eV` gate.
- Full-depth sparse triplet state fit training more strongly, helped the two
  hardest K1-error quintiles, but damaged easier rows and regressed overall.
- Graphormer shortest-path bias also helped hard rows but was strictly worse
  than original PairToken. Globally shared topology bias is therefore closed.
- TGT's transferable causal idea is adjacent pair-to-pair interaction. Its
  dense pair attention, geometry pretraining, model scale, and reported score
  are not imported.

This is the third and final round of the user-authorized sequence. It is not a
retry of either prior topology mechanism.

## Single intervention

At layer 6 only, after the normal K1 edge update and before the local node
block, enumerate the already accepted complete directed non-backtracking
wedges `i -> j -> k` and compute:

```text
EdgeState(i,j) + node(j) + EdgeState(j,k)
                     |
             32-channel message
                     |
        mean return to EdgeState(j,k)
```

The return projection is zero initialized. There is no persistent triplet
state, no direct triplet-to-node return, and no update at the other eight
layers. Original layer-6 PairToken then runs unchanged. The wedge cache is
topology-only and was already accepted for the prior sparse-triplet screen.

The candidate adds no coordinates, bond lengths, angles, dense atom attention,
teacher signal, target residual, prediction fusion, pretraining, or new target.

## Frozen screen

- Accepted cross-platform PCQM-100K V4 rows and graph cache.
- Direct Gap, seed 42, deterministic FP32/no TF32.
- One T4, physical BS128, drop-last, 40 epochs, 31,240 optimizer steps and
  3,998,720 sample presentations.
- AdamW `4e-4`, weight decay `1e-5`, clipping `1.0`, cosine to `1e-6`.
- Immutable K1-v4 reference; no baseline retraining.
- Original PairToken is a retained mechanism control, not a second trained arm.
- Official validation, test-dev, and challenge roles remain sealed.

## Decision rule

Promotion requires at least `0.003 eV` gain over K1 with a favorable paired
interval, plus a non-regression attribution against original PairToken. A
smaller directional result is retained but does not authorize another seed,
scale bridge, or full handoff. Any failure closes this bounded sequence.
