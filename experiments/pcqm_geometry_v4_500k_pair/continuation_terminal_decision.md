# Geometry V4 500K continuation terminal attribution

Date: 2026-09-30. Both continuation arms reached 60 epochs under the
unchanged V4 recipe and passed `experiments/pcqm_500k_v4_evidence/accept_stage.py`.
The authoritative per-arm V5 finalizations are in each continuation directory's
`rml_finalized/`; the paired candidate analysis is
`continuation_pair_analysis.json`.

## Mechanical closure

The physical Kaggle1 kernel was
`nothingnessvoid/molgap-geometry-v4-500k-resume-s42-v1`. Its terminal record
binds source commit `555775ce8fd14966a3566b0e9ff732725279df9f`, source archive
SHA256 `8bea0dab3b8b1755352f7f11e7ae7708b87de0731fe32aefabb9725e60112367`,
and dataset manifest SHA256
`630d30046d6cdc1f91fb169cd1eb4720bd5b352dc1ebeb641a11ead1ebae9751`.
Each arm observed 234,360 optimizer steps and 29,998,080 sample presentations.
The 60-row training traces, selected model, aligned 50K prediction artifact,
last checkpoint, runtime certificate and calibration are locally hash-bound.
Official validation, test-dev and test-challenge remained untouched.

| Arm | Best dev MAE (epoch) | Cumulative T4 device time | Cap | Final stage wall time |
| --- | ---: | ---: | ---: | ---: |
| GPTrans distance-only | 0.1060104072 eV (53) | 12.751797 h | 16 h | 2,974.542 s |
| K1 distance-angle | 0.1048631519 eV (48) | 20.289507 h | 26 h | 28,379.379 s |

These are measured T4 device-hours and per-stage wall-seconds, kept as separate
native measurements. The continuation-only observed costs are recorded beside
each continuation trajectory. Queue hours and CPU hours remain unknown.

The immutable RML continuation cost event for GPTrans has a 0.971-second
overstatement: it subtracted the last trace observation's device counter rather
than the original accepted stage-manifest counter. The exact cumulative total
above is taken from the terminal manifest and remains under the cap. The
finalization transaction refuses changed inputs, so its canonical cost event
remains unchanged; this discrepancy is an explicit record-quality blocker.

## Frozen comparison result

Both candidate prediction files contain 50,000 finite rows with matching
source indices and identical targets. The frozen individual-arm threshold is a
point gain of at least 3 meV against each accepted 500K reference. GPTrans
improves by only 0.857 meV against its reference and misses that threshold.
K1 is 0.0033 meV worse than its reference point MAE and also misses. Its
accepted reference prediction artifact (accepted SHA256
`68fba785a0b028a8445fe8d94d348df2c3a8d7d8caab708acb574d13cac9831b`) is not
locally available, so no K1 reference-paired interval is claimed.

The predeclared fixed 50:50 blend has development MAE `0.1002900898 eV`, a
4.573 meV improvement over the better individual candidate (K1). Its 10,000-draw
paired-row bootstrap 95% percentile interval for blend-minus-K1 is
[-4.937, -4.212] meV. The blend also improves 5.720 meV over GPTrans, interval
[-6.082, -5.365] meV. These intervals quantify row resampling on this fixed
development cohort, not training-seed variability. The result is positive
nomination evidence for the fixed blend under the protocol, and supports a
separate scale review. It does not qualify either arm for replay or authorize
scale-up, production adoption, or protected-role evaluation.

## Failure-mode attribution and disposition

- **Contract, source, data, runtime, artifacts:** the source archive, dataset,
  arm recipes, runtime, role seals and exact terminal exposures match the
  frozen continuation contract. Both mechanical V4 stage acceptances and both
  independent RML terminal pipelines passed.
- **Absolute and paired effects:** neither candidate passes its individual
  reference point gate. The fixed blend has material paired complementarity
  against both candidates on the aligned development cohort. K1-versus-reference
  row-paired uncertainty remains unavailable because the accepted K1 reference
  predictions are missing.
- **Training behavior and endpoint:** each arm reached the frozen 60-epoch
  endpoint and selected its best raw development checkpoint at the epochs in
  the table. The retained trace records live train/development metrics and
  optimizer/presentation axes. This single seed and endpoint do not identify
  seed variability or justify an underfit/overfit diagnosis.
- **Roles and native cost:** only the official-train-derived 500K membership
  and internal 50K development selection roles were used. Both total T4 costs
  stayed within the separately authorized caps. Sealed role reads are false.
- **RML and replay:** RML rebuild and validation succeeded. Both arm traces are
  explicitly excluded from replay-ready qualification because strict matched
  reference pairing is incomplete and the fixed blend is only a nomination
  pending separate scale review. No `READY_FOR_DESKTOP` package was generated.

Terminal disposition: close the continuation as `INCONCLUSIVE` for strict
per-arm reference qualification, while retaining the fixed-blend positive
nomination result. Outstanding blockers are the unavailable accepted 50K K1
reference prediction artifact and the immutable GPTrans RML cost event's
0.971-second overstatement. A separate scale-review decision is required
before any successor. Preserve this experiment branch pending those
reconciliations; do not route the mixed result to production or archive as a
wholly negative scientific result.
