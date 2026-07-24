# Parallel Broad Acquisition Launch (2026-07-21)

Round 01 was downloaded and validated before launch: four CSVs, exactly 60,000
rows, and every SHA-256 matched its published manifest. It is mounted as the
private checkpoint dataset `nothingnessvoid/molgap-2m-broad-round01`.

| Round | Kaggle kernel | Source shard | Target | Launch status |
|---:|---|---:|---:|---|
| 02 | `nothingnessvoid/active-molgap-broad-candidate-fetch-r02-a` | 0 / 2 | 60,000 | Accepted: 60,000 |
| 03 | `nothingnessvoid/active-molgap-broad-candidate-fetch-r03-b` | 1 / 2 | 60,000 | Accepted after reconciliation: 59,997 |

Both tasks exclude the complete 1.5M table, the prior repair union, and every
round-01 CSV. The stable source-file partition is applied before random window
selection, so the two tasks cannot scan the same Hugging Face source file.

Both manifests and CSV hashes were retrieved. Round 03 contained three rows
emitted by both its aromatic and balanced workers; they were removed from the
balanced CSV in the accepted copy. The accepted checkpoint is
`nothingnessvoid/molgap-2m-broad-round02-03`. Round 04 must mount and exclude
this dataset and round 01.
