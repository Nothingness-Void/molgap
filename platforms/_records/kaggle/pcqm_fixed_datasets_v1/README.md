# Kaggle1 Fixed PCQM4Mv2 Dataset Evidence

These private Kaggle datasets are derived only from the official OGB-LSC
PCQM4Mv2 training role and are licensed CC BY 4.0:

- `nothingnessvoid/pcqm4mv2-ogb-fixed-100k-v1`
- `nothingnessvoid/pcqm4mv2-ogb-fixed-500k-scnet-v1`

Only 100K and 500K are permitted on Kaggle1. The 1M and full identities remain
on IMS. The latest dataset versions contain graph shards plus an exact manifest;
raw CSV gzip files were deliberately excluded because Kaggle rewrites their
paths. Every graph contains its source index and target label.

The first uploaded versions are not accepted because Kaggle rewrote compressed
CSV paths. The graph-only latest versions supersede them without deleting the
version history. `acceptance.json` records the accepted latest-version evidence.
IMS export job `1484940.ccpbs1` generated the final parent-linked manifests.
