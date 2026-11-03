# K1 decay-timescale 100K endpoint — 2026-11-03

The exact Kaggle2 kernel `kaseichou/molgap-k1-decay-timescale-100k-s42-r3`,
ID142433892/version1, completed both authorized arms. Both arms passed
mechanical verification with the observed protected-role records untouched.
This record makes one measured, single-seed statement and no causal claim about
weight decay.

## Lineage

The first attempt (`k1-decay-timescale-100k-s42-v1`) and the second
(`k1-decay-timescale-100k-s42-r2`) are infrastructure outcomes. The first failed
because the training worker did not bind the frozen float32 target identity. The
second trained both arms but its canonical traces omitted `learning_rate` and
`checkpoint_identity`, so a strict comparison was impossible. Both defects were
fixed in the infrastructure before this rerun. The science, arms, seed 42,
preset, initialization and roles were unchanged across all three attempts.

## Endpoint

Both arms completed the preset horizon of 40 passes and 31,240 optimizer
updates, returned 0 from preflight and training, and wrote 40 trace observations
each with `learning_rate` and a distinct `checkpoint_identity`. The ledger
records 6,395 s wall and 3.5529 allocated T4-hours.

| Arm | Selected development MAE, meV |
|---|---:|
| reference (weight decay 1e-5) | 141.2945 |
| wd1e3 (weight decay 1e-3) | 141.1010 |

Candidate minus reference selected MAE was -0.1934 meV; the 2,000-draw paired
row bootstrap 95% interval was [-1.0989, +0.6901] meV and the candidate was
better in 66.7% of draws. The final-live delta was -0.1357 meV and the
last-five-observation mean delta was -0.2607 meV. The reference value equals the
value measured for the same frozen initialization in the earlier attempt, so
the run is reproducible on this platform.

## Gate and outcome

The frozen protocol required a selected-MAE lead of at least 3 meV, a paired-row
bootstrap upper bound below 0, a negative final-live delta and a negative
last-five mean delta. The final-live and last-five conditions held, but the
lead is 0.19 meV against the required 3 meV and the interval includes 0, so the
material-lead condition did not pass.

The reference arm closes `CLOSED`. The wd1e3 arm closes `INCONCLUSIVE`: the
difference is not distinguishable from zero in this single-seed screen, and it
is not recorded as evidence against decay, because a row interval measures
molecule-level uncertainty and not training-seed uncertainty. The training
noise level of this setup is a separate, later measurement and was not used to
change the frozen gate.

## Not claimed

No claim is made that weight decay helps or fails to help, no seed
variance estimate, no ranking against other K1 recipes, and no scale-up. The
protected roles were not used, and no extra seed, 500K run or full training was
authorized or run.
