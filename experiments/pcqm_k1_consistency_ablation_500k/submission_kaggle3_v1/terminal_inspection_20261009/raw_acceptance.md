# Complete training / raw comparison - 2026-10-09

Scope: accepted retained-artifact inspection and saved-prediction analysis,
not model inference, clean-BN calibration, adoption or full terminal/RML closure.
Authority: [inspection](inspection_report.json), [paired analysis](paired_selected.json),
[retrieval](retrieval_manifest.json), [scheduler](scheduler_snapshot.json),
[native costs](evidence/invocation_cost.json) and frozen [protocol](../../protocol.md).

## Mechanical acceptance

Kaggle3 kernel `nvoid912/molgap-k1-consistency-500k-pair-s42-v1`,
ID137710959/version1 is COMPLETE. API source bytes match the submitted source;
package, Spec and source identities match the retained release. Both workers
exit0 and preserve the full accepted46-epoch trace prefix exactly.

Both arms finish60epochs,234360 optimizer updates and29998080 presentations.
Selected/resume artifacts match producer SHA256. Model/optimizer tensors and
predictions are finite; checkpoints retain optimizer/RNG and complete-epoch
cursors. Native T4 FP32 certificates and all-arm optimizer preflight pass with
the pinned software identity. Original data/scientific contracts and initialization
are unchanged. Predictions align exactly on internal development rows500000:550000,
with equal targets. Protected evaluation roles remain unconsumed by this attempt.
Numbered epoch predictions were not downloaded.

## Raw selected comparison

| Arm | Selected epoch, one-based | Selected Gap MAE eV | Last epoch MAE eV |
|---|---:|---:|---:|
| coefficient0, mean2 |49|0.105022893|0.105566390|
| coefficient0.1, consistency |49|0.104904077|0.105724610|

Metrics above use retained predictions with CPU float64 recomputation; producer
float32 metrics agree within1e-7. Shared `paired_metrics` checks row/target
alignment and uses1000 paired-row bootstrap draws with seed42. Its unrelated
3meV helper label is not used: this question's frozen material threshold is1meV.

Consistency's selected gain is0.000118816eV,95% row interval
[-0.000419261,+0.000623117]eV;50.38% of rows favor it. The point improvement is
below0.001eV and the interval includes0: the raw nomination gate fails.
The last epoch instead favors coefficient0 by0.000158221eV. Do not substitute
last-epoch metrics for the predeclared selected-checkpoint endpoint.
One seed and repeatedly used development roles limit interpretation; row
bootstrap does not estimate training-seed variability.

## Costs and disposition

Migration training:5.159237 wall-hours,10.318475 allocated T4 device-hours.
Cumulative version1+2+Kaggle3 training:45.160327 T4 device-hours under52;
remaining6.839673 is not authorization for another run. This attempt's full
bootstrap window is5.214331wall-hours/10.428661T4device-hours and includes
training: never add both totals. Queue and CPU-core costs remain unknown.

Disposition: TRAINING_COMPLETE_RAW_COMPARISON_ONLY. Raw results do not establish
a material penalty benefit; no nomination, scale-up or model adoption follows.
The original identical selected-state clean-BN analysis and independent per-arm
V5/RML terminal closures remain pending. Neither trajectory is declared training
replay-ready by this inspection. No continuation or other GPU work was submitted.

## Attribution / remaining discriminator

Source/data/initialization/exposure/resume/native-runtime checks reveal no
mechanical mismatch explaining the weak raw difference. This same-init,
two-forward pair isolates the coefficient, not two forwards versus one forward.
The best endpoints coincide at epoch49, while selected and last orderings differ;
the raw evidence does not demonstrate a stable material benefit of the penalty.
It does not establish that the penalty is harmful or explain general100K-to500K
transfer failures. Train-objective MAE includes the penalty and is not interchangeable
with a clean label-only MAE: no underfitting/overfitting claim is made from it.
Predeclared identical BN calibration is the missing discriminator for buffer
sensitivity versus learned-state differences; causal mechanism remains unresolved.
