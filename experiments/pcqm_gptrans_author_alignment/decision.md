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

The source-layout-only repair in physical v2 passed independent
`results/acceptance.json` after both fixed train shards, deterministic row
samples, per-shard output chunks and native CPU cost were retrieved and
verified. Of 40,559 connected sampled atom pairs at distance 2–20, 38,905
belonged to within-molecule distance groups with multiple bond-path signatures
(95.9%); 26,917 sampled pairs' selected shortest path had a non-single first
bond category. This proves available input distinctions under a deterministic
BFS tie rule, not identity with the author's Cython tensors and not any MAE
benefit. G1/G2 were not released by this CPU result alone.

The subsequent no-training GPU prelaunch check is recorded in
`results/gpu_prelaunch_gaps.json`. The historical GPTrans 100K best checkpoint
and aligned development-prediction files remain locally present and match
their completion-manifest SHA-256 values, but no recovered V5 reusable
reference bundle was present. G1 lacked a frozen transformed initial state;
G2 lacked a full accepted path sidecar and path-tie/sentinel policy. Neither
T4 arm was submitted or treated as a failed scientific test.
