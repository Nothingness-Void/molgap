# K1 InducedPairToken 100K decision

Kaggle1 kernel `nothingnessvoid/molgap-pcqm-k1-induced-pair-s42` version 1
completed all 40 epochs. Candidate-only no-inference acceptance verified the
fixed data identity, source/archive identity, exact K1 initialization,
mechanism checks, 31,240 optimizer steps, 3,998,720 sample presentations,
recoverable checkpoint state, finite aligned 50K development payload, and all
12 completion hashes.

| Model | Parameters | Development MAE | Difference |
|---|---:|---:|---:|
| Frozen K1-v4 | 3,658,817 | `0.1413736343 eV` | reference |
| InducedPairToken | 3,681,921 | `0.1399443895 eV` | `-0.0014292449 eV` vs K1 |
| Original PairToken | 3,681,665 | `0.1383300573 eV` | `-0.0016143322 eV` vs induced |

The induced approximation retained a directional improvement over K1, but it
did not clear the prospectively frozen `0.003 eV` material gate and lost more
than half of the original PairToken gain. The exact aligned K1 prediction bundle
is not currently available locally, so no new paired bootstrap was claimed.
That limitation cannot reverse the failed absolute material gate.

The result is `NEGATIVE_UNDER_CONTRACT`. Bounded induced-pair compression is
closed without another seed, 500K bridge, protected-role access, full training,
or desktop handoff. The scientific attribution is that replacing explicit
cross-node pairs with 16 induced pairs preserved some relation signal but
discarded enough molecule-specific pair structure to remove the material win.

Measured Kaggle kernel wall time was approximately `1.2651 T4-hours`; mean
training throughput was `888.94 graphs/s`. No production model changed.
