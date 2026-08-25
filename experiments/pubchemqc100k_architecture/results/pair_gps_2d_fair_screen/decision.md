# PairGPS2D Fair Validation-Screen Decision

Decision date: 2026-08-25

## Question

The experiment tested whether one pure-2D PairGPS2D encoder could beat the
fixed GPS7-192 plus GPS9-192 equal-prediction comparator on the frozen
PubChemQC-100K validation role under a matched training and selection contract.

## Evidence acceptance

All three learning-rate configurations for both architectures completed with
seed 42, FP32, batch 64, the same 40-epoch schedule, the same accepted 18-wide
graph cache, and the same frozen split. Every retained metric file reports
`test_role_read=false`. The first PBS allocation ended at its wall while the
last PairGPS2D configuration was in progress; the single atomic resume
allocation reused completed configurations and finished the remaining epochs.

The six architecture-level metric files and all six component metrics are
retained under `remote_metrics/`. `validation_summary.json` is the compact
machine-readable comparison. The immutable remote copies, training log, exact
code, and PBS scripts are also preserved in the IMS snapshot referenced by
`../../../../platforms/_records/ims/README.md`.

## Result

Validation selected learning rate `2e-4` for both architectures. PairGPS2D
reduced validation average MAE from `0.1567168579` to `0.1302994364 eV` and
Gap MAE from `0.1980200038` to `0.1556518791 eV`. The deltas were
`-0.0264174215 eV` average and `-0.0423681247 eV` Gap.

## Decision

PairGPS2D passed this validation-only replacement stage. The frozen 9,997-row
test role remained unread, so this record did not authorize a repaired-2M run,
an official PCQM4Mv2 run, or a production change. Later architecture selection
and resource allocation are governed by `CURRENT_STATE.md` and `ROADMAP.md`;
this dated result remains evidence for the isolated PairGPS2D branch.

