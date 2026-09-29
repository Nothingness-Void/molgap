# GPTrans author/local matrix — prelaunch boundary

Decision date: 2026-09-29. The prelaunch evidence `results/prelaunch.json`
identified the accepted Kaggle2 100K cache and the completed synthetic/Cython
parity experiment. Those facts were sufficient to define a label-free real-graph
CPU input preflight, but not an accuracy conclusion or GPU training release.

The P0 preflight samples deterministic training-source rows, reads no Gap
labels, and is separately verifiable by fixed manifest/shard hashes. G1/G2
require their own prospective training identities and inputs after P0 and
code-level gates; O1 changes the optimizer/target contract and therefore
cannot use the old reference as a strict comparator. No result in this record
establishes an MAE contribution or author-score reproduction.

The first physical CPU attempt returned an infrastructure error at graph
deserialization, before any sampled row was read; its immutable receipt is
`results/submission_v1.json`. Its partial `summary.json` reported no labels,
checkpoint, or evaluation role. It is not a failed path hypothesis.
