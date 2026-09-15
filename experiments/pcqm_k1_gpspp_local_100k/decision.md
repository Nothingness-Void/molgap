# Release decision — 2026-09-15

Kaggle2 kernel `kaseichou/molgap-k1-gpspp-local-s42` version 3 completed both
seed-42 arms under the frozen PCQM-100K v4 contract. Saved-artifact acceptance
passed without model inference; official validation, shadow and every test role
remained unread. Source commit was
`8139a0a866e1d169b7d53b91693556fd665059b9` and source archive SHA-256 was
`7aa77876dfbd6a76735c4581e830fcf9e634abdeef69e1bd9064617433e1e223`.

| Arm | Parameters | Best epoch | Development Gap MAE | Gain vs K1 | Paired error-delta bootstrap 95% |
|---|---:|---:|---:|---:|---:|
| Frozen K1-v4 | 3,658,817 | 39 | 0.1413736343 eV | — | — |
| Sender-only directional adapter | 4,515,905 | 35 | 0.1429667920 eV | -0.0015931576 eV | [0.0006835381, 0.0025307218] eV |
| Bidirectional directional adapter | 4,515,905 | 39 | 0.1415791661 eV | -0.0002055317 eV | [-0.0006970280, 0.0011565640] eV |

Neither arm passed the required `0.003 eV` material-gain gate. Sender-only was
a statistically supported regression. The equal-parameter bidirectional arm
beat sender-only by `0.0013876259 eV`, showing that incoming aggregation was
important, but remained statistically indistinguishable from and numerically
worse than K1.

The bidirectional arm reduced epoch-39 normalized training MAE from K1's
`0.0772631` to `0.0686850` while development MAE worsened. Its adapters were
active, exactly nested at initialization, gradient-bearing, resume-equivalent
and memory-safe. Therefore lack of optimization or mechanism activation does
not explain the result. The evidence instead supports redundant local capacity:
K1's ResGatedGraphConv and persistent real-bond EdgeState already exchange
both endpoint signals, while nine extra directional return paths increased
parameters by 23.4% and fitted train-specific structure without improving the
held-out role. Sender-only additionally imposed an asymmetric endpoint bias.

The GPS++-style directional-local adapter family was closed. No seed 43/44,
width, placement, schedule, 500K/full scale, shadow or official-role follow-up
was released. This was the third and final scientific round in the explicit
authorization; frozen K1-v4 remained unchanged.
