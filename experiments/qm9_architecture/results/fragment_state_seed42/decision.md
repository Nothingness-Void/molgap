# BRICS FragmentState QM9-30K Decision

Date: 2026-09-09

## Question

The experiment tested whether deterministic BRICS heavy-atom fragments improve
an OGB EdgeState Structural GPS9 backbone when fragment-to-atom exchange is
inserted at layers 3, 6, and 9. The paired control used the same seed, split,
initial shared weights, optimizer, schedule, FP32 precision, and physical batch
size of 128.

## Accepted evidence

The immutable cache contained 30,000 training and 3,000 validation molecules in
17 shards. Independent cache acceptance and a real batch-128 forward/backward
preflight passed before training. No QM9 test role, official PCQM role, or sealed
role was materialized or read.

| Arm | Best epoch | Validation Gap MAE | Final train Gap MAE |
|---|---:|---:|---:|
| OGB EdgeState Structural GPS9 control | 38 | **0.127442 eV** | 0.091553 eV |
| BRICS FragmentState candidate | 39 | 0.128875 eV | **0.089441 eV** |

The candidate-control gain was `-0.001432 eV`; the frozen promotion threshold
was `+0.001 eV`.

## Interpretation

The fragment stream increased fitting capacity but did not improve held-out
generalization. The lower training error and higher validation error are
consistent with overfitting rather than insufficient optimization. At the last
epoch the learning rate was already `1.61e-6`, so extending the same schedule
would not be a justified recovery experiment.

BRICS partitions encode synthetic bond-cut rules, not electronic delocalization.
Their hard fragment boundaries can therefore separate conjugated regions that
matter for orbital gaps. This run rejects only the exact
`brics-heavy-atom-components-v1` state and exchange design; it does not establish
that every learned or electronically informed hierarchical representation is
unhelpful.

## Decision

The candidate failed the paired gate and was not nominated for PCQM-100K
transfer. No additional seed, longer continuation, full-data training, or model
promotion was authorized. The implementation and compact evidence were retained
on the archive line for reproducibility.

Machine evidence is stored under
`platforms/_records/scnet/qm9_fragment_state_474844e/terminal/`.
