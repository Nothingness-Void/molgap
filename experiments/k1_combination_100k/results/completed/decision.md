# K1 pair-token and readout-gain combination — 2026-11-05

The exact kaggle3 kernel `nvoid912/molgap-k1-combination-100k-s42`,
ID142740152/version1, completed both arms (return 0, 40 trace observations each).
This record applies the gate frozen in `protocol.md` unchanged.

## Measurement

| Arm | Selected development MAE, meV |
|---|---:|
| `reference` | 141.2945 (bit-identical to every earlier run of this recipe) |
| `combination` (`k1_pair_token/1` then `k1_readout_scale/1`) | 139.9298 |

Effect (combination minus reference) -1.3647 meV, 2,000-draw paired row bootstrap
95% interval [-2.2542, -0.5241] meV, probability better 0.9995, 50,000 development
rows. The ledger records 6,717 s wall and 3.7318 allocated T4-hours inside the
4.27 T4-hour ceiling; the combination worker ran 2.0% longer than the reference
worker, so the cost risk named in the protocol did not materialize.

## Outcome

The effect is better than the reference and its upper bound is below 0, but it
does not reach the frozen -3.0 meV material-gain threshold, so the gate fails at
its first condition. `reference` closes `CLOSED`; `combination` closes
`POSITIVE_BELOW_GATE`, the label the protocol assigns to a finite result with
effect < 0 and upper bound < 0 that does not satisfy all four conditions.

## Reading and limits

For scale only: the effect is 0.90 paired SD of the init-only noise (paired SD
1.519 meV), above the 0.81 meV readout-gain-alone result and below the earlier
PairToken result of -3.04 meV on the K1-v4 recipe. One initialization and one
seed; the interval is a row bootstrap and does not include initialization
variance. The development cohort is reused for checkpoint selection. The result
does not separate the pair token from the readout gain, as no single-addon arm
was run. No claim is made about 500K, and no promotion or extra seed follows from
this record. Because the effect is below the protocol's 1.5 meV stop point in
magnitude, this combination is closed with no successor.
