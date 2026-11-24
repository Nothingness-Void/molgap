# Decision

2026-11-17 JST. Run `pcqm-gptrans-tokengt-bond-100k-20261117-s42`, attempt `v1`, strictly retrieved and diagnosed.

## Facts
- Metric: selected EMA MAE on the 50K internal development set (`internal_development`, 50,000 rows) of `pcqm_fixed_100k_v1`. This is an internal split of the training data. Official validation and test were not read.
- `pair_update_norm` (same-job control): 139.440 meV.
- `tokengt_bond` (candidate): 140.676 meV.
- Effect (candidate minus control): +1.236 meV. Paired 2000-bootstrap 95% interval: [+0.281, +2.174] meV. Probability candidate better: 0.0055.
- One seed (42), one init per arm.

## Frozen gate (protocol.md)
Candidate minus control must be <= -3.0 meV with 95% upper bound < 0. Observed +1.236 meV with upper bound +2.174 meV. The gate is not met.

## Outcome
- `pair_update_norm`: `CLOSED` (control arm).
- `tokengt_bond`: `NEGATIVE_UNDER_CONTRACT`. At this 100K, one-seed setting the candidate was slightly worse than the control, and the interval excludes zero on the worse side.
- failure_mode: `below_gate`. This is the closest listed value. The candidate did not reach the gate; it also landed on the worse side of the control. No other listed mode fits better.

## Scope
This result covers one seed at 100K on the internal development set. It says nothing about larger scale, other seeds, or the official validation and test sets, which remain unread.
