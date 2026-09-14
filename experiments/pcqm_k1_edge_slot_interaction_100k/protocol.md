# Frozen protocol — K1 edge/slot pairwise interaction

## Question and candidates

Round 1 found that raw real-bond residual storage with normalized context/read
improved K1 by `0.0008903 eV`, below the material gate. Earlier isolated K1
screens found `0.0017627 eV` from removing length-one slot self-attention and
`0.0018263 eV` from uniform normalized return. This round tests whether the
relation-recurrence signal is additive with each distinct global-slot change:

- `neural_atom_k1_edge_context_no_slot_attention`: normalized-context/raw
  edge storage plus removal of slot self-attention; 3,608,897 parameters.
- `neural_atom_k1_edge_context_uniform_return`: the same edge recurrence plus
  uniform slot return; 3,658,817 parameters.

Each arm changes exactly two previously measured mechanisms and is compared
with immutable K1-v4, not with a newly trained baseline. The arms do not combine
with one another. No geometry, all-pair state, new feature, layer, width,
target residual, fusion, teacher or pretraining is added.

## Immutable V4 contract

Use Kaggle2 `kaseichou/pcqm4mv2-ogb-fixed-100k-v1`, manifest SHA
`1b0e8fd579ab1cb86c02e833e7ad284b4af7582b059f912a77853fdccf3ede6d`.
Train `[0,100000)`, select only on `[100000,150000)`. Seed42, physical BS128
per independent T4, strict FP32/noTF32, deterministic algorithms, drop-last,
40 epochs, 31,240 optimizer steps, 3,998,720 presentations, AdamW4e-4,
weight decay1e-5, clip1, cosine1e-6, normalized direct-Gap L1 and minimum-dev
selection remain frozen. Official validation, shadow and all test roles stay
sealed. Runtime calibration, source inventory, atomic checkpoints, RNG resume
equivalence and ten-epoch recovery chunks are mandatory.

## Decision

No arm passes unless gain over K1-v4 is at least0.003eV, paired-bootstrap
upper95% is below zero and memory reserve is at least15%. Acceptance recomputes
50K ordered payload MAE without constructing or executing a model. A passing
arm becomes a mechanism shortlist only. A directional sub-threshold result is
evidence about additivity, not promotion.

Round3 triple composition is released only if at least one pair clearly retains
both parent effects: its gain must exceed the larger isolated parent gain and
be within `0.0005 eV` of the sum of its two isolated gains. Otherwise stacking
closes and round3 must address a materially different evidence-backed
information-flow question, or stop. No seed/scale/full/official work follows.
