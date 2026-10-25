# Tail regularization decision - 2026-10-25

Both arms: NEGATIVE_UNDER_CONTRACT. Close this bounded warm-start question;
no adoption, extra epochs, seed retry, successor or full-data run is authorized.
Strict mechanical/replay publication is owned separately by the acceptance record.

## Verified endpoints

Kaggle1 kernel140639445/version1 completed the frozen two-arm run. Each arm
completed20 new passes,78120 updates and9999360 gradient sample presentations.
Both full50000-row initial predictor checks are VERIFIED with zero prediction
and MAE difference: initial LIVE MAE0.10490408539772034eV.

| Arm / decay | Selected pass | Selected LIVE MAE/eV | Final LIVE MAE/eV |
|---|---:|---:|---:|
| tail_wd1e3 /0.001 |15|0.10582958906888962|0.10644728690385818|
| tail_wd1e2 /0.01 |15|0.10561738163232803|0.10646752268075943|

The wd1e3 selected trace value is0.10582958906888962eV; small differences
between trace FP32 reduction and retained-prediction FP64 MAE are recorded in
[initial comparison](analysis/initial_comparison.json) and do not affect any gate.
The trace is authoritative for training selection; retained aligned errors own
paired comparisons.

## Frozen gates

Stronger decay improves selected paired-row MAE versus weaker decay by
0.212197003meV,95% row-bootstrap gain CI[0.035779896,0.390094219]meV.
This fails the frozen1meV point gate despite a positive conditional row interval.
Stronger decay leads at15/20 paired evaluations; this is a small numerical signal,
not a useful tail nomination and not independent seed evidence.

Against the identically reproduced initial predictor, selected wd1e3 worsens
0.925511675meV,95% change CI[0.669104829,1.183943146]; wd1e2 worsens
0.713314672meV,CI[0.463719424,0.976997945]. Both fail the required1meV
initial-predictor improvement. Intervals use2000 paired-row draws,seed42,
conditional on development-selected checkpoints; no held-out selection correction
or training-seed variability is measured. Development was already consumed.

## Cost, interpretation and custody

Measured Python runtime/allocation window11817.943945242001s on two Tesla T4s:
23635.887890484002 device-seconds,6.565524414 device-hours total. This includes
idle allocated devices; queue and provisioning before Python remain unknown.
Per-arm process wall through training endpoint is11432.997036165001s and
11447.988494219s, not additive wall time. Native qualification exposure is separate
from new gradient training counters. The observed window is below12T4h;
no claim of complete platform billing/provisioning measurement is made.

[Attribution](attribution.md) distinguishes observed objective reduction from
unmeasured clean fit/BN effects. The fixed uniform TRAIN and BN endpoint follow-up
is insufficient_evidence because its exact graph inputs are not locally retained;
no different cohort, bulk cache download or new remote job substitutes for it.
This does not alter the observed negative endpoint/gate decision.

Keep full rejected source/history in archive and accepted canonical evidence
available to desktop RML. Required payloads remain under D:/molgap-artifacts.
The existing Luna monitor has paused after its terminal notification. No new
training or protected-role use was performed for this acceptance.

Strict closure is blocked by the omitted TRAIN target digest in the frozen target manifest.
See [exact blocker](analysis/strict_blocker.json). Mechanical/scientific closure
does not establish strict replay readiness; no two-complete-entry claim is made.
