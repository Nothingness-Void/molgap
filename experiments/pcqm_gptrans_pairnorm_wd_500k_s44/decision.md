# Terminal decision: GPTrans PairNorm500K WD, initialization44

Job 142336418/version1. Fixed same-job WD0.10 minus WD0.01; sampler/dropout42.

wd_low closes CLOSED; wd_high closes NEGATIVE_UNDER_CONTRACT.

| Arm | Selected pass | Development EMA MAE/eV | Final EMA MAE/eV |
|---|---:|---:|---:|
| wd_low | 37 | 0.101728721504 | 0.102196447551 |
| wd_high | 52 | 0.102859690237 | 0.103061623871 |

Candidate minus reference: +1.130968733meV; conditional paired-row95%CI [0.584297518,1.685900431]meV (2000 resamples, seed42,50000 aligned rows). Both arms complete60passes/234360steps/29998080 presentations. The frozen <=-3meV gate fails.

The companion initialization shows the same unfavorable direction. Mean of two deltas is0.867623789meV; two initializations do not estimate full training variability. This contrast does not establish a strict improvement over historical WD0.05.

Python allocation ledger: 37238.152434 wall-seconds, 20.687862463 allocated T4-device-hours; includes assigned idle time. Queue/provisioning before Python are unknown. Native and replay qualification are separately owned by strict terminal records.

No adoption, successor, retry, full training or additional protected-role use follows. Attribution: [attribution.md](attribution.md).
