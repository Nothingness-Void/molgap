# K1 sparse-triplet decision

Kaggle3 run `nvoid912/molgap-k1-sparse-triplet-s42:v1` completed all 40
epochs and passed saved-artifact acceptance without model inference. It used
the fixed PCQM-100K V4 contract, direct Gap, seed 42, strict FP32/no TF32,
physical batch 128, and no protected role.

| Model | Parameters | Development MAE | Difference vs K1 |
|---|---:|---:|---:|
| Frozen K1-v4 | 3,658,817 | `0.1413736343 eV` | — |
| K1 sparse triplet | 3,766,001 | `0.1424927413 eV` | `+0.0011191070 eV` |

The paired candidate-minus-reference absolute-error bootstrap interval was
`[+0.0002832465, +0.0019549316] eV`. The candidate improved only `48.932%`
of development rows. It reduced normalized training MAE below K1 at every
late epoch while remaining worse on development at every epoch from 30 through
39. At epoch 39, training MAE was `0.0757025` versus K1 `0.0772631`, while
development was worse by `0.0011191 eV`.

The redistribution was strongly difficulty-dependent. Relative to K1, mean
absolute error increased by `0.03911`, `0.01935`, and `0.00216 eV` in the
three easiest K1-error quintiles, but decreased by `0.01217` and `0.04286 eV`
in the two hardest quintiles. Persistent full-depth triplet state therefore
learned useful corrections for difficult molecules but disturbed already good
K1 predictions more than it helped overall. This is an allocation and
regularization failure, not evidence that triplet topology is absent.

The mechanism is negative under its frozen contract. It is also 28% slower
than the contextual K1 reference (`632.58` versus `812.30 graphs/s`). No retry,
extra seed, scale bridge, protected-role read, or full handoff is authorized.
Any successor must avoid another full-depth state/value channel. In particular,
Graphormer-style shortest-path information is only defensible as a small bias
on an already successful selector, not as uniform path aggregation or another
persistent message stream.
