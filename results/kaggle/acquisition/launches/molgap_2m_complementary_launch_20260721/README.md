# Complementary Rare Acquisition Launch

Two CPU-only Kaggle kernels scan disjoint PubChemQC source-file shards while
the exact-2M pure-2D expert trains on SCNet:

| Round | Source shard | Target | Kernel |
|---:|---:|---:|---|
| 06 | 0/2 | 60,000 | `nothingnessvoid/active-molgap-complementary-rare-fetch-r06-a` |
| 07 | 1/2 | 60,000 | `nothingnessvoid/active-molgap-complementary-rare-fetch-r07-b` |

Both kernels exclude the complete 1.5M table, repair union, residual round 01,
and every accepted broad/general/hard acquisition dataset. The four collection
groups are high-Gap rigid/hetero, hetero-dense and multi-S/halogen, bridged or
rigid fused, and conjugated donor-acceptor chemistry.

These outputs are candidate inventory only. Accept counts only after both
kernels terminate, outputs are downloaded, cross-shard CID/canonical-SMILES
deduplication passes, and labels satisfy `gap = lumo - homo`.

## Completion

Round 06 completed with 60,000 rows. Round 07 ended with a transient HTTP
`IncompleteRead`; strict recovery accepted 54,500 durable rows. After complete
prior-inventory and cross-source deduplication, the publishable checkpoint has
114,413 rows. See
`../molgap_2m_complementary_completed_20260721/README.md`.
