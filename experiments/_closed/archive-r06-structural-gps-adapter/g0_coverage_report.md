# Phase 8 Structural GPS Adapter -- G0 Coverage Audit

Input: `results\phase8\gps_arch_dualgps_common_eval_predictions.csv`

This is an analysis-only audit. The routed-v4 Gap prediction is reconstructed
from the stored v3 and dual-GPS predictions using its fixed `4` eV
route rule; no checkpoint or fitted model was used.

## Data contract

- rows: 1977
- valid SMILES: 1977
- invalid SMILES: 0
- finite structural-feature rows: 1977
- routed-v4 Gap MAE: 0.121896 eV
- routed rows: 540 / 1977

## Feature prevalence by routed-v4 Gap-error decile

Decile 10 is the highest absolute Gap-error group.

| error_decile | n | gap_mae | has_ring | atom_in_ring_fraction | smallest_ring_size | max_ring_membership_count | has_fused_ring_atom | fused_ring_atom_fraction | ring_bond_fraction | has_conjugated_bond | conjugated_bond_fraction |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 198 | 0.0055 | 0.9899 | 0.6271 | 5.2677 | 1.4848 | 0.4545 | 0.0514 | 0.5965 | 0.9949 | 0.6515 |
| 2 | 198 | 0.0170 | 0.9798 | 0.6264 | 5.2323 | 1.4848 | 0.4747 | 0.0494 | 0.5942 | 0.9899 | 0.6588 |
| 3 | 197 | 0.0305 | 0.9594 | 0.6190 | 5.1878 | 1.4061 | 0.4264 | 0.0417 | 0.5847 | 0.9848 | 0.6515 |
| 4 | 198 | 0.0457 | 0.9798 | 0.6189 | 5.2576 | 1.4646 | 0.4495 | 0.0475 | 0.5882 | 0.9899 | 0.6572 |
| 5 | 198 | 0.0647 | 0.9899 | 0.6169 | 5.3283 | 1.3939 | 0.3838 | 0.0457 | 0.5848 | 0.9949 | 0.6363 |
| 6 | 197 | 0.0867 | 0.9797 | 0.6267 | 5.2386 | 1.4924 | 0.4873 | 0.0512 | 0.5955 | 0.9949 | 0.6448 |
| 7 | 198 | 0.1168 | 0.9747 | 0.5835 | 5.2121 | 1.3283 | 0.3434 | 0.0332 | 0.5553 | 0.9747 | 0.6035 |
| 8 | 197 | 0.1574 | 0.9594 | 0.5827 | 5.1371 | 1.3807 | 0.4061 | 0.0419 | 0.5566 | 0.9898 | 0.6163 |
| 9 | 198 | 0.2285 | 0.9444 | 0.5482 | 5.0455 | 1.3081 | 0.3434 | 0.0400 | 0.5247 | 0.9798 | 0.5664 |
| 10 | 198 | 0.4657 | 0.9747 | 0.5208 | 5.2222 | 1.3232 | 0.3384 | 0.0316 | 0.5020 | 0.9747 | 0.5611 |

## Highest-decile enrichment

Positive `top_minus_overall` means a structural feature is more prevalent in
the worst routed-v4 Gap decile than in the evaluated population.

| feature | overall_mean | top_decile_mean | top_minus_overall | top_over_overall |
|---|---|---|---|---|
| has_ring | 0.9732 | 0.9747 | 0.0016 | 1.0016 |
| atom_in_ring_fraction | 0.5970 | 0.5208 | -0.0762 | 0.8724 |
| smallest_ring_size | 5.2129 | 5.2222 | 0.0093 | 1.0018 |
| max_ring_membership_count | 1.4067 | 1.3232 | -0.0834 | 0.9407 |
| has_fused_ring_atom | 0.4107 | 0.3384 | -0.0723 | 0.8239 |
| fused_ring_atom_fraction | 0.0434 | 0.0316 | -0.0118 | 0.7281 |
| ring_bond_fraction | 0.5682 | 0.5020 | -0.0662 | 0.8834 |
| has_conjugated_bond | 0.9868 | 0.9747 | -0.0121 | 0.9877 |
| conjugated_bond_fraction | 0.6247 | 0.5611 | -0.0636 | 0.8982 |

## G0 result

Features with positive highest-decile enrichment: has_ring, smallest_ring_size.

This result alone does not authorize fitting. G1 remains conditional on a
chemically interpretable enrichment rather than a feature-frequency artifact,
as specified in `pre_registration.md`.
