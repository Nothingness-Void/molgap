# CPU input preparation v1 failure diagnosis

On 2026-09-30, Kaggle2 preparation kernel
`kaseichou/molgap-gptrans-author-inputs-preparation-v1` (kernel ID 136492377,
version 1) ended with API status ERROR. The downloaded `failure.json` reported
an IndexError in `_rederive_shard`: source-shard position 1000 was used to index
a 1000-row chunk-local `graph_node_count` tensor. The first chunk began at zero,
so its verification did not expose the coordinate mix-up.

The correction used `local_position` for that chunk tensor while retaining
original shard `position` for source graph lookup and source-ID calculation.
Regression coverage included nonzero chunk starts over both training shard
offsets and the internal-development offset, plus rejection of altered node
counts. No path policy, feature, model, training budget or scientific gate was
changed. This local fix did not establish complete real-source rederivation.

The manifest reported construction complete, but execution and independent
acceptance failed. No `preparation_result.json` established successful CPU
verification. Neither the complete flag nor the local regression result
qualified G1/G2 GPU compute. No GPU successor was submitted by this repair.
The retained chunks may support a separately qualified CPU verification
recovery; do not rebuild them or relabel the failed attempt as successful.

Retained original evidence root (ignored storage):
`platforms/_records/kaggle/training/gptrans_author_inputs_preparation_v1/gptrans_author_inputs/`.
Artifacts include `failure.json`, `native_cost.json`, `startup.json`,
`paths/manifest.json`, path chunks, and retrieved frozen source. The physical
release and source identity are in `../preparation_submission.json`.

Reported native cost: 434.295763 wall seconds, 420.192984 process-CPU seconds,
four allocated CPU cores and zero GPUs. The failure record explicitly reported
`training_executed=false` and `model_inference_executed=false`; absent observations
were not inferred. This was an infrastructure failure, not a scientific negative
or a replay-ready training result.
