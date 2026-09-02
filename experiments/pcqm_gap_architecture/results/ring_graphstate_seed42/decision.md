# Ring-GraphState seed-42 decision

On 2026-09-03 Kaggle1 kernel
`nothingnessvoid/molgap-pcqm-ring-graphstate-s42`, version 1, completed both
fresh paired candidates under the frozen contract. The Ring-GraphState
candidate reduced internal-validation Gap MAE by only
`0.00020264089107513428 eV`, added 58,040 parameters, and ran at 0.8492 times
the baseline throughput.

The frozen no-model acceptance did not pass because both candidate
`metrics.json` files omitted `input_cache_aggregate_sha256`. The top-level
selection, checkpoints, validation payloads, source identity, geometry cache,
ring cache, row identity, target identity, and artifact hashes remained
available. This was an output-metadata defect rather than a numerical or model
execution failure, but the experiment's declared advancement gate required
every artifact to pass acceptance.

The result was therefore not advancement eligible. Independently, its very
small paired gain did not justify multi-seed confirmation under the repository
seed-budget rule, especially with a 15.1% throughput loss. The exact
RingState64 addition was closed without a seed, width, depth, schedule, or ring
definition retry. The frozen GraphState9 winner remained unchanged.

The compact arithmetic and provenance are in [`summary.json`](summary.json).
Large checkpoints, payloads, traces, and logs remain under
`platforms/_records/kaggle/training/pcqm_gap100k_ring_graphstate_seed42_v1/`.
Official PCQM validation and test-dev were not read, and no full-data,
desktop, or molecular-research-server action followed from this result.
