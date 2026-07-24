# Candidate acquisition continuation launch

Three CPU-only Kaggle tasks continue acquisition against the complete
1,557,037-row accepted inventory:

| Workload | Target | Kernel |
|---|---:|---|
| Complementary R10, shard 0/2 | 60,000 | `nothingnessvoid/active-molgap-complementary-rare-fetch-r10-a` |
| Complementary R11, shard 1/2 | 60,000 | `nothingnessvoid/active-molgap-complementary-rare-fetch-r11-b` |
| General overnight R03 | 500,000 | `nothingnessvoid/active-molgap-general-overnight-fetch-r03` |

Complementary tasks checkpoint every 250 accepted rows. General R03 uses a new
seed/window schedule and packages five independently retrievable 100K chunks.
Cross-task CID/canonical-SMILES acceptance remains mandatory after completion.
