# Kaggle Acquisition Audit (2026-07-21)

## Conclusion

The remembered 1.5M table exists as `data/raw/phase8_repair_v3_1p5m.csv` and as
`nothingnessvoid/molgap-2m-fetch-base/phase8_repair_v3_1p5m.csv`. It has already
completed the controlled pure-2D gate and is negative. It must not be retrained
or interpreted as a missing 2M acquisition.

No complete labeled 2M table exists in the current Kaggle account. The kernel
named `molgap-2m-candidate-fetch` has produced one durable 60K round toward a
future pool; `2M` is the intended future dataset size, not its current row count.

## Acquisition inventory

| Acquisition | Task status | Raw / recovered rows | Strict unique rows | Training use |
|---|---|---:|---:|---|
| Repair-v2 unchunked fetch | complete | 600,000 quota rows | 550,699 | Included in the 726,966-row repair union |
| Repair-v2 chunked rounds 1-3 | complete | 179,996 | 179,996 | Included in the 726,966-row repair union |
| Repair-v2 strict union | reconciled | 726,966 | 726,966 | Best-ranked 500K formed the negative additive 1.5M run |
| Residual-target round 01 | Kaggle wrapper error; outputs recovered | 45,520 raw | 45,457 labeled | New residual-focused source |
| Future broad round 01 | complete | 60,000 | 60,000 labeled | New mostly balanced source |
| Future broad round 02 | complete | 60,000 | 60,000 labeled | Stable source shard 0/2 |
| Future broad round 03 | complete | 60,000 | 59,997 labeled | Stable source shard 1/2; three cross-bucket duplicates removed |
| General overnight round 01 | complete | 500,000 | 499,471 labeled | Five 100K chunks; 529 prior-pool overlaps removed |
| Hard rounds 04-05 | complete partial; accepted | 145,996 | 144,943 | Residual regions only; 1,053 within-round duplicates removed |

The repair union and additive 1.5M tables have exact HOMO/LUMO/Gap labels, but
their controlled external result is already closed: OOD and PCQM improve while
common-all and P8-hard regress significantly. The remaining approximately 227K
unused repair candidates come from the same rejected acquisition distribution
and are not the next training priority.

## New pilot assembly

The broad 60K and residual 45,457 sources contain 102,798 unique rows after
CID/canonical-SMILES deduplication. Scaffold keys were computed for all original
1M rows and all candidates using resumable 50K atomic chunks.

- Candidate rows on scaffolds absent from original 1M: 60,315.
- New held-out set: 5,000 rows / 3,381 whole scaffolds.
- The held-out scaffolds are absent from both original 1M and the top-up.
- Uniform training top-up: 97,798 rows.
- Final training table: 1,097,798 rows with the original 1M as an exact prefix.
- No replay weighting is used.

Assembly evidence: `results/phase8/broad_residual_pilot/assembly_report.json`.

Separately, future broad rounds 01-03 now provide 179,997 strict unique rows.
They are acquisition inventory only and were not part of the 1.098M pilot above.
Rounds 02-03 are published as the private checkpoint dataset
`nothingnessvoid/molgap-2m-broad-round02-03`.

The accepted general overnight run raised this acquisition inventory to 679,468
strict unique rows. Hard rounds 04-05 add 144,943 strict rows after source
exhaustion and reconciliation, bringing the complete inventory to 824,411.
The accepted hard rows have zero overlap with all prior acquisition checkpoints
and are published as
`nothingnessvoid/molgap-2m-hard-round04-05-accepted`.

## Training gate

SCNet jobs:

| Job | Stage |
|---:|---|
| 699293 | Build and validate appended 2D graph cache; completed |
| 699294 | Uniform GPS7 continuation from original 1M checkpoint; completed |
| 699295 | Uniform GPS9 continuation from original 1M checkpoint; completed |
| 699296 / 699762 | Dual-GPS 2D head; killed before script start by SCNet signal 53 |
| 699298 / 699763 | Paired external evaluation; cancelled by failed dependency |

The equal-cost original-1M control is job 698900 and completed successfully.
Candidate encoders and embeddings are complete, but no comparable candidate
head or external metrics exist yet. Promotion still requires a stable Gap
improvement of at least 0.001 eV without a significant P8-hard or PCQM
regression. No 3D compute is allocated before this gate passes. Current SCNet
blocker: `results/phase8/broad_residual98k_scnet/README.md`.
