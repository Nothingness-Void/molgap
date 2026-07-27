# Parallel Hard-Chemistry Acquisition Launch (2026-07-21)

| Round | Kaggle kernel | Source shard | Target | Startup status |
|---:|---|---:|---:|---|
| 04 | `nothingnessvoid/active-molgap-hard-candidate-fetch-r04-a` | 0 / 2 | 100,000 | Complete partial; 72,599 accepted |
| 05 | `nothingnessvoid/active-molgap-hard-candidate-fetch-r05-b` | 1 / 2 | 100,000 | Complete partial; 72,344 accepted |

Both CPU-only tasks mount the complete exclusion base and all three accepted
acquisition checkpoints, covering 679,468 strict candidate rows. They scan only
the fixed residual-analysis regions:

- very-large high-sp3, non-aromatic, or flexible molecules;
- very-large macrocycles or multi-amide molecules;
- flexible molecules with B3LYP Gap 2.5-4.0 eV;
- high-sp3 non-aromatic molecules.

No balanced-general rows are collected. Each task uses 24 windows per source
file and writes independent group CSVs, atomic progress, reports, hashes, and a
ZIP. Source exhaustion may produce a valid partial result below 100K.

Both results were retrieved and strictly reconciled. See
`results/kaggle/acquisition/completed/molgap_2m_hard_parallel_completed_20260721/README.md`.
