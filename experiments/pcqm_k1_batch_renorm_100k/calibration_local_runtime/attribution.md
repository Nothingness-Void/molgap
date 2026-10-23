# Selected-LIVE buffer attribution

| Arm | Original MAE eV | Clean-BN MAE eV | Change meV | Reproduction maximum error eV |
|---|---:|---:|---:|---:|
| brn_scale | 0.1416963935 | 0.1460725814 | +4.3762 | 7.629394531e-06 |
| brn_affine | 0.1433418989 | 0.1601097137 | +16.7678 | 2.861022949e-06 |

Candidate-minus-reference gap rises from+1.645509meV to+14.037127meV;
calibrated paired-row95% bootstrap interval is[13.059133,14.981390]meV
(2000draws,seed42). Both calibrated endpoints worsen. This rejects the narrow
hypothesis that this same-prefix cumulative Clean-BN procedure removes the
mean-correction disadvantage at the selected LIVE checkpoints.

The candidate is more sensitive to this buffer replacement. Persistent learned
function differences and weight/statistics co-adaptation remain compatible with
the observation; it does not isolate which training-time mechanism caused them.
Clean-BN is an intervention, not a guaranteed oracle for population statistics:
fixed-prefix composition, cumulative batch estimation and dropout-off mode can
also shift the evaluation state. Do not infer that all BN calibration fails,
that all BatchRenorm variants fail, or that ordinary BN is superior from this pair.

The full training curves show identical first5passes, divergence after mean
correction activates, and no late relative recovery over40passes. Together with
this diagnostic, extra exposure or buffer repair is unsupported as the next
remedy for this exact affine variant. No baseline was retrained and no posthoc
calibrated state is selected for promotion. Training-seed variability remains
unknown; paired row uncertainty is not seed uncertainty.

Reuse: matching family loader, FamilyTrainingState.calibrate/evaluate,
CalibrationPolicy, BoundForeachEMA holding an unchanged selected-LIVE snapshot,
shared forward/layout/evaluation and paired bootstrap; existing RML finalize.
No actual EMA updates or trained EMA-weight substitution occurred.

Actual role use:100Ktrain+50Kdevelopment labels decoded for frozen input/target
normalization validation; first16384train features for calibration with labels
unused in its objective;50Kdevelopment predictions and metrics, already consumed
for selection. Official validation/test and other protected roles untouched.

Runtime cost:77.1906021s wall,244.546875s process CPU,68.5893883s one-local-GPU
allocation window; distinct units, no T4/A100 conversion. Preparation and the
separately closed failed attempt001 are excluded from these execution costs.
