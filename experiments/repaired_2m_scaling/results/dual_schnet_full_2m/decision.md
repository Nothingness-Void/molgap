# Repaired-2M Dual-SchNet Completion

## Decision

The repaired-2M Primary and Augmented lightweight SchNet branches completed
under the frozen `176/160/6`, cutoff-10-A protocol. Both checkpoints and all
source-aligned embedding parts passed the independent acceptance contract.
They were authorized as inputs to the single predeclared bounded 2D+3D Fusion
gate; this record did not promote either standalone SchNet or change the
production registry.

## Accepted Metrics

| Branch | Best validation average MAE | Test average MAE | Test HOMO | Test LUMO | Test Gap |
|---|---:|---:|---:|---:|---:|
| Primary | 0.120544 | 0.120416 | 0.108540 | 0.106866 | 0.145843 |
| Augmented | 0.127352 | 0.127012 | 0.115950 | 0.109593 | 0.155493 |

All values are in eV. The Primary run used BF16 stable recovery from the finite
FP16 epoch-7 state and selected recovery epoch 4. The Augmented run trained
under BF16 and selected epoch 7. Neither accepted run recorded a non-finite
batch.

An independent local evaluation of the Primary best state reproduced
validation/test average MAE `0.120580/0.120435 eV`, within `0.00004 eV` of the
remote backend. A separate Colab recovery reported `0.623343 eV` test MAE while
retaining `best_epoch=-1`; it contradicted the full 198,925-row local
recomputation of the same source state (`0.132068 eV`) and was rejected as an
environment or evaluation-path anomaly.

## Embedding Acceptance

- Primary parts: 100
- Augmented parts: 100
- Aligned rows: 1,989,116 of 2,000,000
- Coverage: 99.4558%
- Embedding dimension: 176
- Inference coordinates: primary conformer for both branches
- Source identity, targets, finite tensors, dimensions, part counts, and
  per-file SHA256: accepted

## Evidence

- `primary_metrics.json`
- `augmented_metrics.json`
- `embedding_acceptance.json`

The sealed 20K was not accessed.
