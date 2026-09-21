# Frozen protocol: K1 PairToken + MoSE

## Question

Can the accepted layer-6 PairToken benefit from a complementary rooted motif
structural view without prediction fusion or a second encoder?

## Evidence basis

- PairToken is the only retained K1 relation mechanism that cleared the fixed
  100K materiality gate.
- MoSE31 replacement was directionally positive but below gate, and its saved
  residuals are not redundant with PairToken.
- A saved-prediction-only diagnostic found residual correlation `0.82877`,
  absolute-error correlation `0.80140`, and MoSE lower absolute error on
  `49.058%` of development rows. A 50/50 prediction average reached
  `0.1328849941 eV`; this is hypothesis evidence only and is not the candidate.
- Triplet, SPD, gating, and persistent-state variants are closed under this
  contract, so this round tests a different structural-view composition.

## Single intervention

The candidate is the exact accepted PairToken parent plus one node-level
MoSE31 residual before the nine graph blocks:

```text
OGB atom features + RWSE16 + zero-return MoSE31 residual
                    |
          K1 local/slot graph blocks
                    |
        unchanged layer-6 PairToken
                    |
             direct Gap head
```

MoSE31 passes through `31 -> 64 -> 192`; the final projection is zero
initialized. PairToken is unchanged. There is one encoder and one prediction
head: no prediction fusion, gate, coordinates, geometry, teacher, pretraining,
target residual, or protected-role access.

## Frozen screen

- Accepted cross-platform PCQM-100K V4 graph cache plus accepted MoSE31 sidecar.
- Direct Gap, seed 42, deterministic FP32/no TF32.
- One T4, physical BS128, drop-last, 40 epochs, 31,240 optimizer steps and
  3,998,720 sample presentations.
- AdamW `4e-4`, weight decay `1e-5`, clipping `1.0`, cosine to `1e-6`.
- Immutable K1-v4 reference; no baseline retraining.
- Original PairToken is the parent attribution control, not a trained arm.

## Decision rule

Promotion requires at least `0.003 eV` gain over K1 with a favorable paired
interval and a favorable paired non-regression interval against PairToken.
A smaller directional result is retained but authorizes no seed or scale run.
A negative result closes PairToken+MoSE composition under this contract.
