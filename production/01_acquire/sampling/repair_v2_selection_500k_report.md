# Phase 8 Repair-v2 500K Selection

This artifact is the new 500K top-up only. It does not alter or concatenate the frozen expansion500K base.

- strict candidate union: 726,966
- usable after exclusion: 726,966
- selected top-up: 500,000
- base-overlap rows in selection: 0
- rejected-1M overlap rows in selection: 0
- selected unseen base scaffolds: 427,130

## Method

- Strict CID/canonical-SMILES union in documented source order.
- Explicit exclusion against frozen expansion500K and rejected expansion1M.
- Bucket quotas scaled from the 600K collection specification to the 500K selection target.
- Within each bucket: unseen Bemis-Murcko scaffold, lower base-scaffold frequency, lower candidate-scaffold frequency, then stable SHA-256 tie-break.
- This is a scaffold-novelty selection. It does not claim an exhaustive all-pairs Morgan-fingerprint nearest-neighbour calculation.

## Bucket Audit

| bucket | requested | available | quota selected | top-up selected | shortfall |
|---|---:|---:|---:|---:|---:|
| `very_low_gap` | 4,167 | 6,470 | 4,167 | 0 | 0 |
| `low_gap_aromatic_edge` | 4,167 | 6,463 | 4,167 | 0 | 0 |
| `large_aromatic_edge` | 25,000 | 34,855 | 25,000 | 0 | 0 |
| `very_large_general` | 16,667 | 25,539 | 16,667 | 0 | 0 |
| `s_or_cl_hard` | 33,333 | 45,401 | 33,333 | 0 | 0 |
| `aromatic_edge_general` | 25,000 | 38,674 | 25,000 | 0 | 0 |
| `flexible_hard` | 25,000 | 28,005 | 25,000 | 0 | 0 |
| `large_mw_500_700` | 41,666 | 60,565 | 41,666 | 0 | 0 |
| `balanced_general` | 325,000 | 480,994 | 325,000 | 0 | 0 |
