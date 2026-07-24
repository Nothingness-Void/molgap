# PCQM GINE Expert Pilot Decision

The Kaggle version 3 run completed and its artifacts passed integrity
acceptance. All declared artifact and graph-shard SHA256 values match the
downloaded evidence. The 11 shards contain 254,997 unique, finite-label graphs
from 255,000 source rows; three invalid molecules were explicitly excluded.
The train, scaffold-development, and fixed official-validation counts are
229,335, 20,662, and 5,000.

The best model reached 0.212343 eV Gap MAE on the scaffold-disjoint development
split and 0.213504 eV on the fixed official-valid 5K. This improves the routed
v4 reference by 0.078186 eV, but misses the predeclared 0.20 eV scale gate by
0.013504 eV.

Decision: reject this checkpoint as the PCQM specialist prerequisite for the
hierarchical Oracle study. Do not train a learned Router or expand GPS9/fusion
from this result. The checkpoint remains reproducibility evidence and a useful
benchmark-specific baseline. Official PCQM test splits and the future sealed
20K were not accessed, and the production registry is unchanged.

Machine-readable acceptance: `acceptance.json`.
