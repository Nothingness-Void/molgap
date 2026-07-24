# Retention-D Seed 42 Decision

## Decision

Retention-D seed 42 passes the predeclared general-model gate and advances to
the two fixed repeat seeds. It is not promoted to production before those
repeats finish.

## D Minus Retention-B

| Scope | Average MAE delta (eV) | Gap MAE delta (eV) |
|---|---:|---:|
| Common | -0.001163 | -0.000382 |
| OOD-1000 | -0.001184 | -0.000124 |
| P8-hard | -0.001141 | -0.000645 |
| PCQM valid 5K | n/a | +0.000502 |

The one-week seed-42 gate covers common, OOD, and P8-hard. All three average
MAEs improve, and both hard domains exceed the required `0.001 eV`
improvement. PCQM is recorded as a specialist-domain regression and remains
outside this general-model gate.

The training metric `init_compatible=false` denotes strict full-state loading
rather than partial compatible loading. The requested warm start was loaded;
this field is not an initialization failure.

## Follow-up

- SCNet `707264` trains seed 43; dependent `707265` evaluates it.
- SCNet `707266` trains seed 44; dependent `707267` evaluates it.
- The split seed remains 42. Only model, sampler, and training-order seeds
  change.
- GPS9, fusion, sealed-set access, and production registration remain blocked
  until all three GPS7 seeds point in the same direction.

Exact metrics and artifact hashes are in
`retention_d_seed42_comparison.json`.
