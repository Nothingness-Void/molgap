# K1 decay-timescale second attempt — 2026-11-04

The exact Kaggle2 kernel `kaseichou/molgap-k1-decay-timescale-100k-s42-r2`,
ID142334233/version1, completed both arms. This attempt closes as
`INFRASTRUCTURE_ONLY` for both arms. It is superseded by the third attempt, whose
[terminal decision](../completed_r3/decision.md) owns the measurement.

## Why infrastructure only

Both arms trained the 40-pass horizon and returned 0, but their canonical traces
omitted `learning_rate` and `checkpoint_identity` in all 40 rows, so a strict
causal comparison could not be established. The omission was an infrastructure
defect in how the training session recorded trace rows, fixed before the third
attempt. It cannot be repaired afterwards because the fields are written during
training. Nothing about the science, arms, seed, preset, initialization or roles
differed between this attempt and the third.

## What is recorded

Raw development observations from this attempt are unaccepted: `reference`
141.2945 meV and `wd1e3` 141.1010 meV. They equal the third attempt's values, so
the run is reproducible on this platform, and they are not used as evidence. The
ledger records 6,422 s wall and 3.5675 allocated T4-hours.

## Not claimed

No acceptance, no comparison class, no effect estimate, no Replay admission and no
claim about weight decay is made from this attempt. Protected roles were not used.
