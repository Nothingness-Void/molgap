# Decision

2026-10-30: terminal closure of the bounded initial-residual 100K pair under [protocol](protocol.md). The
numbers below are the ones in [decision_draft.md](decision_draft.md), which also holds the mechanical facts,
the descriptive readouts and the attribution notes.

- Development Gap MAE: baseline 0.1412945 eV, initial_residual 0.1420021 eV. Gain (baseline minus candidate)
  is -0.708 meV. Paired row bootstrap (2000 resamples, seed 42) gives a candidate-minus-baseline error of
  +0.708 meV, 95% interval [-0.170, +1.551] meV. The gate needed gain >= 3.0 meV with a lower bound above
  zero, so it fails by 3.7 meV.
- Outcome: `initial_residual` is `NEGATIVE_UNDER_CONTRACT`. The same-job `baseline` is `CLOSED` as the
  reference. One seed; row uncertainty does not measure training-seed variation. No retry, seed, 500K or
  adoption follows.
- Protected roles: `untouched` for official_validation, test_dev and test_challenge, as structural evidence
  (`role_audit_v2/`). It is not an observation of file opens; it states that the protected roles were not
  among this run's declared mounted inputs.
