# Protocol: K1 MoSE hidden normalization on fixed PCQM 100K

## Question

Does the normalization used inside the published MoSE count encoder correct
the complexity-dependent regression observed when MoSE31 replaced RWSE16 in
K1-v4?

The prior K1-MoSE experiment is closed under its own question. This is a new,
isolated implementation-fidelity question released by post-run trajectory and
subgroup evidence, not a seed, width, schedule, or threshold rescue.

## Evidence and single change

- The accepted K1-MoSE run improved K1-v4 by `0.0011191219 eV`, but missed its
  prospectively frozen `0.003 eV` gate.
- Across its final 15 epochs, MoSE averaged `0.0009701014 eV` below the matched
  K1 trace, so longer training is not the supported repair.
- In the lowest MoSE-magnitude quintile, its rowwise absolute-error delta was
  `-0.0031019719 eV`; in the highest quintile it reversed to
  `+0.0016426733 eV`.
- The published PCQM MoSE configuration uses a two-layer count MLP with hidden
  BatchNorm and does not enable raw-input BatchNorm. The prior MolGap encoder
  used two linear layers with SiLU but no normalization.

The candidate therefore changes exactly this module:

```text
previous: log1p(MoSE31) -> Linear(31,192) -> SiLU -> Linear(192,192)
candidate: log1p(MoSE31) -> Linear(31,192) -> BatchNorm(192)
                            -> SiLU -> Linear(192,192)
```

It reuses the accepted immutable MoSE cache. RWSE is still replaced, not
concatenated. K1 local EdgeState blocks and layer-3/6/9 single-slot exchanges
are unchanged. No geometry, teacher, target-derived feature, prediction
fusion, or protected role is permitted.

## Frozen V5 screen

- Reference: immutable `neural_atom_k1_v4` accepted payload.
- Diagnostic predecessor: accepted `neural_atom_k1_mose` payload.
- Candidate: `neural_atom_k1_mose_hidden_bn`, expected 3,662,081 parameters.
- Fixed official-train-derived PCQM roles: 100,000 train / 50,000 development.
- Seed 42; deterministic FP32; TF32 disabled; physical BS128; `drop_last`.
- AdamW `4e-4`, weight decay `1e-5`, gradient clip `1.0`.
- Cosine schedule, 40 epochs, identical row order and sample exposure.
- Official validation, test-dev and test-challenge remain unread.

## Decision gate

Mechanical acceptance requires complete artifacts, hashes, a deterministic
runtime certificate, the accepted MoSE cache identity, and equality of every
non-intervention contract field. Scientific promotion requires at least
`0.003 eV` improvement over immutable K1-v4 and a favorable paired row
interval. The old MoSE payload is reported only to determine whether the
normalization repair itself helped.

A pass grants shortlist status only. A miss closes this normalization question
without another seed, 500K, protected role, or normalization variant.
