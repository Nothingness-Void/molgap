# Decision

2026-10-30: terminal closure of the bounded extremal-readout 100K pair under [protocol](protocol.md).
The numbers below are the ones in [decision_draft.md](decision_draft.md), which also holds the
mechanism readout and attribution.

- Development Gap MAE: baseline 0.1412945 eV (pass 40), candidate 0.1446276 eV (pass 39, step 30459).
  Gain (baseline minus candidate) is -3.333 meV. Paired row bootstrap (2000 resamples, seed 42) gives a
  candidate-minus-baseline error of +3.333 meV, 95% interval [+2.303, +4.320] meV. The gate needed gain
  >= 3.0 meV with a lower bound above zero, so it fails, and the interval excludes zero in the harmful
  direction.
- Outcome: `candidate` is `NEGATIVE_UNDER_CONTRACT`. The same-job `baseline` is `CLOSED` as the
  reference. One seed; row uncertainty does not measure training-seed variation. No retry, variant, seed
  or scale-up follows.
- Protected roles: `untouched` for official_validation, test_dev and test_challenge, as structural
  evidence (`role_audit_v2/`). It is not an observation of file opens; it states that the protected roles
  were not among this run's inputs.
