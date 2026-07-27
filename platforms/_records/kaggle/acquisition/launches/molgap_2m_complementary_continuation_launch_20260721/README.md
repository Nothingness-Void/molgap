# Complementary acquisition continuation

Two CPU-only Kaggle kernels continue the rare-region scan using disjoint source
file shards and a new round seed:

| Round | Source shard | Target | Kernel |
|---:|---:|---:|---|
| 08 | 0/2 | 60,000 | `nothingnessvoid/active-molgap-complementary-rare-fetch-r08-a` |
| 09 | 1/2 | 60,000 | `nothingnessvoid/active-molgap-complementary-rare-fetch-r09-b` |

Both mount the complete 938,824-row accepted acquisition inventory as
exclusions. They checkpoint every 250 accepted rows and package each group
independently. Outputs remain candidates until terminal retrieval and strict
CID/canonical-SMILES reconciliation.
