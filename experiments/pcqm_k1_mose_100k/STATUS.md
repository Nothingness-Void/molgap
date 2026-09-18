# Status

Implementation and V5 protocol are committed and pushed on `molgap-server`.
Private Kaggle1 source dataset
`nothingnessvoid/molgap-pcqm-k1-mose-source` is `ready` at source commit
`12c62e476e174e9f204977e218fbf85334958045`.

Kaggle1 CPU kernel `nothingnessvoid/molgap-pcqm-k1-mose-cache-prep` version 1
failed before graph loading because the CPU image did not provide
`torch_geometric`. This was an infrastructure failure, not a scientific
result. No model was trained and no sealed role was read.

The only covered retry pinned the pure-Python `torch-geometric==2.6.1` wheel by
SHA-256 and completed as version 2. Local no-model acceptance passed all
identity, row-count, source-range, shard-hash, aggregate-hash and sealed-role
checks. The accepted cache has 150,000 rows, 2,090,775 nodes and 31 channels;
its aggregate SHA-256 is
`5d949f90a35aea5001d2f6438c916f92cab45d6c5860ba6ccf5777aac1c897e3`.
The immutable private dataset is
`nothingnessvoid/molgap-pcqm-k1-mose-cache-v1`. Acceptance evidence is
`results/cache_acceptance_v2.json`.

After confirming that no other recent Kaggle1 GPU training remained active,
the single authorized seed-42 screen
`nothingnessvoid/molgap-pcqm-k1-mose-s42` version 1 was submitted and entered
`RUNNING`. No additional seed, protected role, 500K job or successor is
authorized by this submission.
