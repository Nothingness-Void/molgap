# QM9 Charge-Adapter Screen Protocol v2

## Question

Does a deterministic continuous partial-charge prior improve the accepted OGB
EdgeState Structural GPS9 without adding another message-passing hierarchy?

The preceding BRICS FragmentState candidate fit the training role better but
regressed validation by 0.001432 eV. That result closes hard fragment
partitioning. This experiment instead adds information at the atom input and
keeps every downstream layer unchanged.

## Frozen comparison

| Field | Control | Candidate |
|---|---|---|
| Backbone | OGB EdgeState Structural GPS9 | same |
| Added input | Gasteiger charges | same |
| Adapter | 2 -> 16 -> 192, frozen at zero | same, trainable |
| Target | direct QM9 Gap | same |
| Split | 30,000 train / 3,000 validation, seed 42 | same |
| Model seed | 42 | same |
| Physical batch/device | exactly 128 | same |
| Precision | FP32 | same |
| Training | AdamW, 40 epochs, patience 8 | same |

Charges use RDKit Gasteiger iteration count 12. Each channel is standardized
with train-only atom statistics. `LayerNorm(1)` is intentionally forbidden:
normalizing a scalar independently would erase it.

Both arms execute the identical charge-adapter computation graph. The sham
control permanently freezes its adapter at its bitwise-zero initialization;
the candidate trains it. All state tensors must be bitwise equal before the
first update and both adapter outputs must be bitwise zero. Cross-instance
prediction difference is retained as a DCU diagnostic, not used as a proxy for
computational equivalence. A real accepted-cache batch of 128 must pass finite
forward/backward on SCNet before training starts.

This v2 sham-control contract supersedes the earlier comparison against the
adapter-free backbone. That earlier preflight exposed a DCU rounding difference
from the extra zero-add path and stopped before either arm trained; widening its
tolerance would not establish an exact paired comparison.

## Gate and boundaries

PCQM-100K transfer is nominated only when candidate validation Gap MAE improves
by at least 0.001 eV. This is a one-seed architecture triage, not production or
leaderboard evidence. Failure closes this charge-adapter mechanism; it does not
authorize another seed, width, charge algorithm, or schedule retry.

Only QM9 train and internal validation roles are materialized. No QM9 test,
official PCQM validation, test-dev, shadow, sealed data, pretrained weights,
fusion, or production registry is read or changed.

## Durability

CPU output consists of atomic 2,000-graph shards, progress, manifest, and
independent acceptance. Each training arm writes an atomic checkpoint and
trace after every epoch, plus a best checkpoint and aligned validation payload.
The terminal completion manifest hashes every retained artifact.
