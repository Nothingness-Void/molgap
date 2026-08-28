# PubChemQC PairGPS2D Full-Data Decision

Decision date: 2026-08-25

The early repaired-2M PairGPS2D seed-42 job did not establish an accepted
model. Its first submission was too slow to produce an epoch checkpoint; the
throughput-repaired run later reached best validation average MAE `0.474529
eV` and was stopped. Its automatic resume was removed and full training was
not authorized.

This attempt was superseded by the matched PubChemQC-100K validation-only
screen, which compared PairGPS2D against the required GPS7 plus GPS9 equal
control before any long run. That later decision is owned by
`../pubchemqc100k_architecture/results/pair_gps_2d_fair_screen/decision.md`.

No production, official PCQM, or full repaired-2M claim follows from this
directory. Exact job identifiers, cache acceptance, throughput measurements,
and the stop record are in `results_remote_submission.json`.
