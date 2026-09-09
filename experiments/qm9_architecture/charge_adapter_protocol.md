# QM9 Charge-Adapter Screen Protocol

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
| Added input | none | Gasteiger atom and implicit-H charges |
| Adapter | none | 2 -> 16 -> 192, zero-output initialized |
| Target | direct QM9 Gap | same |
| Split | 30,000 train / 3,000 validation, seed 42 | same |
| Model seed | 42 | same |
| Physical batch/device | exactly 128 | same |
| Precision | FP32 | same |
| Training | AdamW, 40 epochs, patience 8 | same |

Charges use RDKit Gasteiger iteration count 12. Each channel is standardized
with train-only atom statistics. `LayerNorm(1)` is intentionally forbidden:
normalizing a scalar independently would erase it.

The candidate's shared tensors must equal the control's at initialization and
its adapter output must be bitwise zero. Predictions must agree within an
absolute tolerance of `1e-6`, allowing the harmless last-bit difference caused
by an explicit zero-add kernel on heterogeneous accelerators. A real
accepted-cache batch of 128 must pass finite forward/backward on SCNet before
training starts.

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
