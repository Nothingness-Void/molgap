# Width256 terminal decision

Date: 2026-10-02 (Asia/Tokyo). Outcome: **NEGATIVE_UNDER_CONTRACT**.

The authorized single candidate completed its fixed 100K, seed42, 40-epoch
screen. Mechanical artifact inspection and actual T4 runtime preflight passed.
The [saved-prediction analysis](kaggle3_reconciliation_v1/scientific_metrics.json)
compares the same 50K ordered internal-development rows and identical targets:

| Model | Parameters | MAE (eV) |
|---|---:|---:|
| Retained width192 reference | 3,658,817 | 0.1412944608205557 |
| Width256 candidate | 6,035,201 | 0.14279833381593227 |

Gain is reference MAE minus candidate MAE: -0.001503872995376587 eV.
The 95% row-bootstrap interval is [-0.0024646552270650864,
-0.0005251574828624728] eV. Both the point gain and lower confidence bound
fail the frozen +0.003 eV material gate. This uncertainty is conditional on
these saved predictions; training-seed variability remains unknown.

## Interpretation and qualification limits

The submitted width change did not improve this screen. Keep the width192
reference. This result does not establish that wider atom vectors are generally
harmful, that width192 loses necessary chemical information, or that edge64 /
slot64 is the bottleneck. Read the [attribution](kaggle3_reconciliation_v1/attribution.md)
for the matched trace observations and unresolved explanations.

Both selected minima occur at epoch40. Their online training metrics include
dropout and differ in cohort/timing from development metrics; they do not
establish overfitting, underfitting or insufficient exposure. No further epoch,
seed, module or scale-up is released by this result.

Both runs qualified their own T4 / FP32 tuple. Core torch, CUDA, cuDNN and
hardware match, but installed distributions differ, including NumPy2.0.2
versus1.26.4. Retain the paired endpoint conclusion without claiming an
isolated architecture speedup or equivalence of the full software environments.
The reference ran in a separate paired job; candidate invocation time is not
a controlled architecture cost comparison.

The prospective contract pins the retained reference bundle, but its canonical
`state_at_start.reference_ids` is empty: the reused reference arm has no
independently accepted V5 evidence ID frozen here. Preserve this gap; do not
retroactively add an evidence ID or rewrite the prospective record. Retain the
complete trace as a hashed artifact and exclude this result from strict RML
replay/backtest admission. Metadata finalization does not certify replay.

## Disposition

Scientific decision: close this width-only 100K question as negative under its
contract. No model promotion or successor is authorized. Retain the owning
branch and durable outputs pending review of terminal Git routing; no desktop
integration, archive transition or master promotion is implied by acceptance.
Reopening requires a new decision-relevant, explicitly authorized contract.
