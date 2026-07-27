# Candidate acquisition continuation acceptance

## Terminal results

| Workload | Raw rows | Accepted rows | Decision |
|---|---:|---:|---|
| Complementary R08-R09 | 120,000 | 119,934 | accepted |
| General overnight R02 | 500,000 | 498,279 | accepted |
| Total | 620,000 | 618,213 | accepted |

R08 and R09 both completed all four groups with matching source hashes. Strict
acceptance removed 26 prior-inventory overlaps and 40 cross-source duplicates.
General R02 completed five independently packaged 100K chunks; rare rows were
given priority, then 1,721 general overlaps were removed.

Across the final 618,213 rows, labels are finite, CID and canonical-SMILES
duplicates are zero, and the maximum `gap - (lumo - homo)` magnitude is
`3.55e-15 eV`. The complete future candidate inventory is now 1,557,037 rows.

Private checkpoints:

- `nothingnessvoid/molgap-2m-complementary-round08-09-accepted`
- `nothingnessvoid/molgap-2m-general-overnight-r02-accepted`

These rows do not change the frozen exact-2M experiment or model registry.
