# K1 Muon100K terminal decision

2026-11-06 JST. Job142927073/version1, Kaggle2. This decision applies only to
the two fixed optimizer recipes in [protocol](protocol.md), not optimizer
families after unrestricted tuning or500K transfer.

Both newly trained arms completed40passes/31240updates/3998720presentations;
selected and final weight state is Live, with best development at pass40.
Saved predictions have50000 aligned development rows. Existing mechanical
inspection reports no per-arm blockers; native/strict replay qualification is
owned by the subsequent shared acceptance report, not this decision alone.

| Arm | Selected prediction MAE/eV | Outcome |
| --- | ---: | --- |
| adamw | 0.14129446082055568 | CLOSED |
| muon | 0.1325206676620245 | POSITIVE_UNDER_CONTRACT |

Candidate-minus-reference=-8.773793159meV; paired2000-resample95%interval
[-9.708100418,-7.861417456]meV. Final trace delta=-8.773804meV; mean of
last-five matched-pass deltas=-8.487302meV. All four frozen nomination conditions
pass. Prediction aggregation versus stored trace float rounding explains the
sub-micro-eV scalar difference; no selected weight-state substitution occurred.
This is one seed and reused development; row uncertainty is not seed variance.

Measured process-wall endpoints: AdamW5499.850894s, Muon11177.999007s,
ratio2.032419. Allocation wall11502.737373s and23005.474747T4-device-seconds
(6.390410T4-device-hours) include both allocated devices and idle time. These
are not GPU-busy or queue measurements. No2x speed or compute-efficiency win
is claimed. Strict artifacts remain on D:; no all-output retrieval occurred.

Nominate this fixed single-model recipe for a separately authorized scale
decision after strict closure. Do not adopt, launch500K/full, retrain a reference,
add a blind optimizer grid or infer guaranteed500K improvement. Preserve both
outcomes, actual costs and [attribution](attribution.md).
