# Gap-Only And Normalized-RWSE Screen Decision

Decision date: 2026-08-24

## Question

This round tested two changes to the accepted three-output RWSE16 Structural
GPS9 encoder on the same immutable PubChemQC 100K scaffold split:

1. replace joint HOMO/LUMO/Gap supervision with a single Gap target;
2. normalize the atom and RWSE branches and learn a bounded scalar RWSE gate.

The comparison used random initialization, seeds 42/43/44, 40 epochs, and the
same `100,003/10,000/9,997` train/validation/test rows. The old three-output
checkpoints were retained unchanged. No common, OOD, P8-hard, official PCQM
test, or sealed labels were read.

## Artifact acceptance

Both Kaggle kernels reached terminal `COMPLETE`. All six new runs passed model,
training-state, metrics, prediction, finite-value, source-index, target, graph,
and split checks. Each prediction file contains exactly 9,997 aligned Gap
targets. The accepted input SHA256 values are:

| Input | SHA256 |
|---|---|
| Base graph | `5c348c28c8f75f09d6072ebf88de28f8513feec6e7a20c8439edc12d4b18d936` |
| RWSE16 graph | `71d61923ea008b02eb902003c73eaf9aee9f6ff488be5f88482f0e11d700e017` |
| Scaffold split | `1e6707274dd8465cfe9d96a808064372af705c4a9e4b8d20532ae6fff2cdcf05` |

Machine-readable acceptance is in `acceptance.json`.

## Results

| Variant | Validation Gap MAE, mean | Test Gap MAE, mean | Three-seed test ensemble | Parameters | Mean training time |
|---|---:|---:|---:|---:|---:|
| Three-output RWSE16 Structural GPS9 | not comparable: checkpoint selected on three-target average | **0.164735 eV** | **0.156568 eV** | 3,776,259 | 35.34 min |
| Gap-only RWSE16 Structural GPS9 | 0.173276 eV | 0.176531 eV | 0.169111 eV | 3,776,065 | 35.88 min |
| Normalized/gated RWSE16 Gap-only | **0.168639 eV** | **0.172089 eV** | **0.163570 eV** | 3,776,450 | 35.76 min |

Gap-only increased test Gap MAE against the three-output model by
`0.012492/0.008958/0.013936 eV` for seeds 42/43/44, with a mean regression of
`0.011795 eV`. Removing HOMO and LUMO supervision was therefore harmful despite
giving the model a target-specific checkpoint criterion.

Normalized/gated RWSE improved Gap-only validation by
`0.003131/0.002136/0.008643 eV` and test by
`0.005082/0.000639/0.007605 eV`. It passed the predeclared local gate of a
direction-consistent validation improvement of at least `0.001 eV` in the
three-seed mean. However, it still regressed against the accepted three-output
model by `0.007353 eV` mean test Gap MAE and by `0.007003 eV` in the equal-seed
ensemble.

The learned RWSE mixing coefficients ended at
`0.249470/0.249266/0.249750`, essentially unchanged from the `0.25`
initialization. The observed gain over Gap-only therefore supports normalized
branch mixing, but does not show that the scalar gate learned meaningful
sample- or seed-specific routing.

## Decision

The Gap-only hypothesis was rejected. Joint HOMO/LUMO/Gap supervision provided
useful auxiliary signal for Gap prediction on this screen.

The normalized/gated RWSE implementation was retained as positive component
evidence but was not promoted as a standalone architecture. Passing its local
comparison against Gap-only did not override its regression against the
three-output Structural GPS model.

The earlier three-output RWSE16 Structural GPS9 remained the only architecture
authorized for one bounded repaired-2M standalone scale-up. This round did not
authorize a second 2M run, production promotion, a registry change, or external
evaluation.
