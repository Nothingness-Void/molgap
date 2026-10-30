# Decision

2026-10-30: terminal closure of the bounded SAM 100K pair under [protocol](protocol.md). The numbers
below are the ones in [decision_draft.md](decision_draft.md), which also holds the cost readout and
descriptive readouts.

- Development Gap MAE: baseline 0.1412945 eV, sam 0.1441285 eV. Gain (baseline minus candidate) is
  -2.834 meV. Paired row bootstrap (2000 resamples, seed 42) gives a candidate-minus-baseline error of
  +2.834 meV, 95% interval [+1.936, +3.754] meV. The gate needed gain >= 3.0 meV with a lower bound above
  zero, so it fails, and the interval excludes zero in the harmful direction.
- Outcome: `sam` is `NEGATIVE_UNDER_CONTRACT`. The same-job `baseline` is `CLOSED` as the reference.
  One seed; row uncertainty does not measure training-seed variation. No retry, rho sweep, LookSAM, seed
  or scale-up follows.
- Protected roles: `untouched` for official_validation, test_dev and test_challenge, as structural
  evidence (`role_audit_v2/`). It is not an observation of file opens; it states that the protected roles
  were not among this run's inputs.
