# Terminal decision: GPTrans PairNorm500K WD, initialization43

Job 142336090/version1. Fixed same-job WD0.10 minus WD0.01; sampler/dropout42.

wd_low closes CLOSED; wd_high closes NEGATIVE_UNDER_CONTRACT.

| Arm | Selected pass | Development EMA MAE/eV | Final EMA MAE/eV |
|---|---:|---:|---:|
| wd_low | 43 | 0.102289038491 | 0.102725856006 |
| wd_high | 53 | 0.102893317337 | 0.103065289557 |

Candidate minus reference: +0.604278846meV; conditional paired-row95%CI [0.065947964,1.157838065]meV (2000 resamples, seed42,50000 aligned rows). Both arms complete60passes/234360steps/29998080 presentations. The frozen <=-3meV gate fails.

The companion initialization shows the same unfavorable direction. Mean of two deltas is0.867623789meV; two initializations do not estimate full training variability. This contrast does not establish a strict improvement over historical WD0.05.

Python allocation ledger: 38486.796899 wall-seconds, 21.381553833 allocated T4-device-hours; includes assigned idle time. Queue/provisioning before Python are unknown. Native and replay qualification are separately owned by strict terminal records.

No adoption, successor, retry, full training or additional protected-role use follows. Attribution: [attribution.md](attribution.md).
