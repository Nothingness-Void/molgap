# EdgeState Convergence Decision

## Question

The isolated continuation tested whether the accepted OGB-rich full-data
EdgeState run had stopped before convergence. It preserved the official graph
cache, split, seed, architecture, and source checkpoint identity defined in
`protocol.md`.

## Accepted result

| Metric | Source epoch 19 | Continued epoch 30 | Delta |
|---|---:|---:|---:|
| Official-validation Gap MAE | 0.102062 eV | **0.099638 eV** | **-0.002424 eV** |

Job `1364434.ccpbs1` completed with exit code zero after 18 additional epochs
and 11.18 hours. Epoch 30 was selected; seven subsequent non-improving epochs
satisfied the fixed patience condition. The relative validation improvement
was 2.37%.

Independent acceptance verified the immutable source hashes, completion
manifest, result hashes, 73,545 unique validation identities, finite model and
prediction tensors, and a locally recomputed MAE of 0.0996382982 eV. The run
used official train and validation only. It did not use official test,
external data, pretrained weights, or the MolGap production registry.

## Decision

The continuation is accepted as positive convergence evidence and its epoch-30
checkpoint is the strongest validation checkpoint for this isolated
OGB-rich EdgeState line. The original 20-epoch run was underconverged under its
initial schedule. Further continuation with the same schedule is closed because
validation failed to improve for seven epochs after epoch 30.

This is a training-convergence result, not an architecture comparison. It does
not modify the submitted OGB form, public reproduction release, official test
artifacts, or Track A production model. Any new official inference or
submission requires a separate explicit decision.

Machine-readable acceptance is in `acceptance.json`; retrieved remote evidence
is preserved beside this record.
