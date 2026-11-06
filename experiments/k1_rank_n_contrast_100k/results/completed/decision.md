# K1 Rank-n-Contrast 100K pair — 2026-11-06

The exact kaggle3 kernel `nvoid912/molgap-k1-rank-n-contrast-100k-s42`, ID 142996802
version 1, completed both arms on T4x2 (40 trace observations each); both passed strict
mechanical verification. This record applies the gate frozen in `protocol.md` unchanged.

## Measurement

| Arm | Selected development MAE, meV |
|---|---:|
| `reference` | 141.2945 (bit-identical to every earlier run of this recipe) |
| `rnc` (`k1_rank_contrast/1`, lambda 0.007, tau 0.1) | 140.2361 |

Effect (rnc minus reference) -1.0584 meV, 2,000-draw paired row bootstrap 95% interval
[-1.9834, -0.1525] meV, probability better 0.987, 50,000 development rows. Final-live
delta -1.0340 meV; mean delta over the last five observations -1.1697 meV. The ledger
records 6,619 s wall and 3.68 allocated T4-hours inside the 4.27 T4-hour ceiling.

## Outcome

The upper bound is below 0 and the final-live and last-five deltas are negative, but the
effect does not reach the frozen -3.0 meV material gate, so condition 1 fails.
`reference` closes `CLOSED`; `rnc` closes `POSITIVE_BELOW_GATE`, the label the protocol
assigns to effect < 0 and upper bound < 0 that does not satisfy all of 1-4. The effect is
under the protocol's 1.5 meV stop point, so Rank-n-Contrast closes at 100K with no
tau, lambda, projection-head, normalisation, two-view or RankSim successor.

## Reading and limits

Observed: the paired difference by pass was -7.8, -8.2 and -10.4 meV at passes 5, 10 and
15 and shrank to -2.1, -3.9, -1.8, -1.8 and -1.0 meV at passes 20 to 40, so the early lead
faded (the saved-prediction diagnosis also reports `late_horizon_fade`). The rnc arm's
final training L1 was lower (0.0721 versus 0.0770 normalised). Interpretation, not
observation: a regulariser that helps early and fades late is consistent with the joint
scheme being the weakest in the paper and with the reference catching up, as in the
joint atom reconstruction result; this study does not isolate why. The effect is 0.70
paired SD of the init-only noise (1.519 meV) and the row interval does not include
initialization variance. One initialization and one seed; development rows were reused for
selection. Cost: the two training workers' wall times are within about 1% of each other
(per-arm attribution not separately verified here); the T4 step-time ratio was not
isolated. No claim is made about 500K, other lambda or tau, or the embedding-geometry
mechanism: the Spearman and clip-share probes named in the protocol were not run.
No protected role was used and 3,658,817 parameters were unchanged (training-only term).
