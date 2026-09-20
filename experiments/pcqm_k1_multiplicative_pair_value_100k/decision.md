# Multiplicative PairValue decision

Kaggle3 run `nvoid912/molgap-k1-multiplicative-pairvalue-s42:v1`
completed all 40 epochs and passed saved-artifact acceptance without model
inference. It used the fixed PCQM-100K V4 contract, direct Gap, seed 42,
strict FP32/no TF32, physical batch 128 and no protected role.

| Model | Parameters | Development MAE | Difference vs K1 |
|---|---:|---:|---:|
| Frozen K1-v4 | 3,658,817 | `0.1413736343 eV` | — |
| Original PairToken | 3,681,665 | `0.1383300573 eV` | `-0.0030435771 eV` |
| Multiplicative PairValue | 3,694,081 | `0.1424374580 eV` | `+0.0010638237 eV` |

Against K1, the paired candidate-minus-reference absolute-error bootstrap
interval was `[+0.0001260218, +0.0019842576] eV`. Against the retained
original PairToken endpoint, Multiplicative PairValue was worse by
`0.0041074143 eV`, with interval `[+0.0032307103, +0.0049826595] eV`.
The regression is therefore directional at the paired-row level and not a
material-gain near miss.

The separate ordered Hadamard value did not expose useful conjunctions under
the accepted additive learned-query selector. It instead removed the stable
additive relation content that carried the original PairToken gain while
adding parameters and reducing throughput from the contextual K1 reference's
`812.30` to `763.83 graphs/s` across accepted runtimes.

This candidate is negative under the frozen contract. PairToken selector/value
micro-architecture enrichment is closed: chemistry conditioning, node-adaptive
return and multiplicative value all failed to improve the original PairToken.
No retry, extra seed, scale bridge, protected-role read or full handoff is
authorized. A future experiment must test a distinct information-flow family.
