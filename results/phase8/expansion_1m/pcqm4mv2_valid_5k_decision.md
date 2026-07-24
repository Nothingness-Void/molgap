# PCQM4Mv2 Public Valid 5K Check

This is a local, deterministic 5,000-row sample from the official PCQM4Mv2
public `valid` split (73,545 rows), not an OGB leaderboard submission and not
the hidden test-dev/test-challenge score. It measures only the official
`homolumogap` label. Nineteen molecules failed ETKDG graph construction, so the
paired comparison contains 4,981 molecules.

| model | Gap MAE (eV) | Gap R2 | delta vs routed-v4 |
|---|---:|---:|---:|
| routed-v4 | 0.291690 | 0.828432 | baseline |
| 1M candidate | 0.304685 | 0.808710 | +0.012995 |

The paired bootstrap 95% CI for 1M minus routed-v4 Gap MAE is
`[+0.009411, +0.016737] eV`; the candidate is worse on this public-validation
check. This reinforces the P8.13 decision: retain the 1M candidate only as a
documented targeted-hard specialist, not a global default.

The 5K sample is selected from official valid with `random_state=42` and its
source indices are stored in `pcqm4mv2_valid_5k_predictions.csv`.
