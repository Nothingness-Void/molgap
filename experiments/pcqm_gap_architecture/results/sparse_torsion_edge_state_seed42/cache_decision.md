# Sparse Torsion Cache Decision

Decision date: 2026-08-31

## Acceptance

The separate Kaggle1 CPU derivation completed in 276.99 seconds. Its output
was downloaded to the durable record root and passed
`accept_pcqm100k_torsion_cache.py` without model inference. All 100,000 train
rows and 10,000 internal-validation rows were retained; the 315 invalid parent
geometry rows retain their paths with zero features and zero masks. Official
validation and test-dev roles were not read.

## Result

- Torsion cache aggregate: `4e151e4d11cf84c53e2510c2982670736a91503899f6e2dc27efedf220bb0173`
- Train paths: 4,627,762; valid train paths: 4,608,590
- Validation paths: 462,972; valid validation paths: 461,492
- Degenerate paths masked: 20,652
- Private account1 mirror: `nothingnessvoid/molgap-pcqm-torsion-cache-s42-dataset`

## Decision

The immutable sparse torsion cache is accepted as the sole input for the one
seed-42 paired GPU screen. The GPU runner remains bounded to its frozen
distance-plus-angle comparator and torsion candidate contract.

## Evidence

- No-model acceptance: [`cache_acceptance.json`](cache_acceptance.json)
- Compact summary: [`cache_summary.json`](cache_summary.json)
- Remote identity: [`cache_launch_manifest.json`](cache_launch_manifest.json)
