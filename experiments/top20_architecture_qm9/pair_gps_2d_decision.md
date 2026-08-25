# Pair-GPS pure-2D preflight decision record

## First complete attempt

Job `1318042.ccpbs1` ran the first `pair_gps_2d` implementation on the fixed
30,000/3,000/3,000 QM9 split with encoder seed 42. It completed 20 epochs on
one A100 and wrote its checkpoint, model, payload, log, and metrics. The model
had 4,707,939 parameters and used no conformer or old prediction.

Its fixed-test metrics were:

| metric | pair-GPS attempt | GPS-only pure-2D reference | delta |
|---|---:|---:|---:|
| HOMO MAE (eV) | 0.1051761 | 0.1018309 | +0.0033452 |
| LUMO MAE (eV) | 0.1137967 | 0.1067166 | +0.0070801 |
| Gap MAE (eV) | 0.1466797 | 0.1414196 | +0.0052602 |
| average MAE (eV) | 0.1218842 | 0.1166557 | +0.0052285 |

The candidate therefore failed the strict same-contract architecture gate.
The failure is an accuracy result, not a runtime or stability failure. The
first implementation's custom bond-local aggregation did not retain the
standard GPS/GINE local branch, and its capacity was substantially below the
two-encoder reference. No full QM9, seed-43/44, or PubChemQC B3LYP training
was submitted.

## Isolated structural refinement

The retained direction is the same single pure-2D Pair-GPS encoder. The
retry3 implementation increases the fixed encoder capacity to a 256-channel,
10-layer, 96-channel pair stack and restores a true bond-local `GINEConv`
whose edge state is read from the current persistent pair state. The pair
state still supplies global-attention bias, all-pair-to-node messages, and
low-rank triplet mixing. This is an encoder architecture change, not a target
residual, output fusion, warm start, or 3D addition. Its preflight output is
isolated under `outputs/results_pair_gps_2d_retry3`.

## Completed FP32 refinement and gate decision

After retaining the failed AMP attempts separately, job `1318345.ccpbs1`
completed the fixed refinement in FP32 on one IMS A100. It used the same
30,000/3,000/3,000 split, split seed 42, and encoder seed 42 as the frozen
Route A/B-style reference. The direct three-target head used no old
predictions, residual target, fusion, warm start, coordinates, or conformer.

| metric | PairGPS2D refinement | GPS9 + GPS11-160 fusion | delta |
|---|---:|---:|---:|
| HOMO MAE (eV) | 0.0957187 | 0.1018309 | -0.0061122 |
| LUMO MAE (eV) | 0.1055230 | 0.1067166 | -0.0011936 |
| Gap MAE (eV) | 0.1340952 | 0.1414196 | -0.0073244 |
| average MAE (eV) | 0.1117790 | 0.1166557 | -0.0048767 |

Lower is better, so the refinement passed both predeclared primary gates:
average and Gap. It did not beat the optional, more heavily fused secondary
reference (average 0.0974363 eV, Gap 0.1171055 eV), which was never the
replacement gate because it is not the requested two-encoder Route A/B-style
comparator.

This is an architecture-selection result, not a QM9 leaderboard claim. It
authorizes one target-domain PubChemQC B3LYP seed-42 training run of the same
single PairGPS2D encoder. It does not authorize residual learning, model
fusion, 3D input, fine-tuning, or an ensemble.
