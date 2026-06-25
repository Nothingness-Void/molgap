# P8.2 Sampling Spec

Purpose: fill sparse Phase 7 training regions before retraining a v2 B3LYP base.

- current training rows analyzed: 300,000
- planned targeted top-up: 200,000
- keep Phase 7 300k as base distribution; do not redraw another same-source 300k
- hard filters: elements subset of C/H/N/O/S/F/Cl, MW 200-1000, gap > 0, exclude Phase 7 CIDs/canonical SMILES

## Current Gap Summary

| region | n | fraction |
|---|---:|---:|
| gap_lt_2p5 | 912 | 0.304% |
| gap_lt_3 | 4,041 | 1.347% |
| gap_3_to_4 | 42,633 | 14.211% |
| high_conjugation | 50,442 | 16.814% |
| aromatic_rings_ge_5 | 5,045 | 1.682% |
| aromatic_fraction_ge_0p8 | 6,204 | 2.068% |
| aromatic_edge | 6,639 | 2.213% |
| mw_ge_500 | 19,637 | 6.546% |
| mw_ge_700 | 3,285 | 1.095% |
| has_s | 98,198 | 32.733% |
| has_cl | 61,092 | 20.364% |
| has_s_or_cl_hard | 25,443 | 8.481% |
| flexible_hard | 9,405 | 3.135% |

## Priority Fetch Buckets

Assign each fetched candidate to the first matching bucket with remaining quota.

| priority | bucket | quota | current n | current fraction | predicate |
|---:|---|---:|---:|---:|---|
| 1 | `very_low_gap` | 30,000 | 912 | 0.304% | `gap < 2.5` |
| 2 | `low_gap_aromatic_edge` | 40,000 | 533 | 0.178% | `2.5 <= gap < 3.2 and (aromatic_rings >= 5 or aromatic_atom_fraction >= 0.85)` |
| 3 | `large_aromatic_edge` | 26,000 | 2,889 | 0.963% | `mw >= 500 and (aromatic_rings >= 5 or aromatic_atom_fraction >= 0.85)` |
| 4 | `very_large_general` | 20,000 | 3,285 | 1.095% | `mw >= 700` |
| 5 | `s_or_cl_hard` | 20,000 | 25,443 | 8.481% | `(has_s or has_cl) and (gap < 3.5 or aromatic_rings >= 4 or aromatic_atom_fraction >= 0.70)` |
| 6 | `aromatic_edge_general` | 18,000 | 5,975 | 1.992% | `gap >= 3.2 and (aromatic_rings >= 5 or aromatic_atom_fraction >= 0.85)` |
| 7 | `flexible_hard` | 10,000 | 9,405 | 3.135% | `rotatable_bonds >= 8 and (gap < 3.5 or aromatic_rings >= 4)` |
| 8 | `large_mw_500_700` | 36,000 | 16,352 | 5.451% | `500 <= mw < 700` |

## Axis-Level Desired Coverage

These are diagnostics for the final old+topup pool; priority bucket quotas above are the executable fetch plan.

| axis | current n | current fraction | desired final fraction | needed in top-up |
|---|---:|---:|---:|---:|
| gap_lt_3 | 4,041 | 1.347% | 8.0% | 35,959 |
| gap_lt_2p5 | 912 | 0.304% | 3.0% | 14,088 |
| high_conjugation | 50,442 | 16.814% | 10.0% | 0 |
| aromatic_rings_ge_5 | 5,045 | 1.682% | 6.0% | 24,955 |
| aromatic_fraction_ge_0p8 | 6,204 | 2.068% | 6.0% | 23,796 |
| mw_ge_500 | 19,637 | 6.546% | 18.0% | 70,363 |
| mw_ge_700 | 3,285 | 1.095% | 5.0% | 21,715 |
| has_s_or_cl_hard | 25,443 | 8.481% | 12.0% | 34,557 |
| flexible_hard | 9,405 | 3.135% | 8.0% | 30,595 |

## Next Step

Implement a targeted PubChemQC skim/fetcher that computes these cheap descriptors before graph building,
fills the priority buckets, writes a slim CSV, then holds out scaffold-disjoint hard eval slices per bucket.
