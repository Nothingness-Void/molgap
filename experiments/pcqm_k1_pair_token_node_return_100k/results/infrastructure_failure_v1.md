# Kaggle3 version 1 infrastructure failure

Kernel `nvoid912/molgap-k1-node-adaptive-pairtoken-s42` version 1 reached a
Tesla T4 and verified the frozen source archive, then failed before cache
loading with `Expected one fixed 100K cache, found []`.

Kaggle rejected the Kaggle2-private dataset source when version 1 was pushed,
so the fixed cache was absent from the run. No graph was loaded, no optimizer
step ran, no development label was read, and no protected role was accessed.
This is an infrastructure failure, not a scientific result.

Kaggle3 already owns `nvoid912/pcqm4mv2-ogb-fixed-100k-v1`; its five listed
files have the exact accepted sizes, including the three immutable shard
sizes. Version 2 changes only the dataset owner in kernel metadata. Source,
cache hash checks, model, seed, precision, batch, optimizer, schedule, exposure
and roles are unchanged.

