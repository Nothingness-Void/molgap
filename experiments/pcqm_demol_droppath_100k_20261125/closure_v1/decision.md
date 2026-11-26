# DeMol-CIN stochastic depth 100K decision

2026-11-26 JST. `no_droppath` closes `CLOSED`. `droppath010` closes `POSITIVE_UNDER_CONTRACT` under the [frozen protocol](../protocol.md).

| Arm | Best EMA MAE (meV) | Best pass | Pass-60 EMA MAE (meV) | Give-back (meV) |
|---|---:|---:|---:|---:|
| `droppath010` | 131.195 | 43 | 132.188 | 0.992641 |
| `no_droppath` | 134.635 | 35 | 138.426 | 3.791004 |

The frozen comparison is `droppath010` minus `no_droppath`. The aligned pass-60 estimate is **-6.238 meV**, paired 95% interval **[-7.168, -5.314] meV** over the same 50,000 internal development rows. The main EMA checkpoint rescores matched the remote pass-60 trace within the scorer's 0.0001 eV validation tolerance. Give-back uses the unrounded native trace. The training trace has 60 passes, 46,860 updates and 5,998,080 presentations per arm. The measured allocation ledger records 10.967 T4-device-hours across both arms, including allocated idle time. Queue time was unmeasured.

DropPath .10 passes the endpoint gain, paired interval and give-back gates. This closes the 100K experiment and nominates the candidate for scale-transfer review.

[Endpoint rescoring, file pins and paired results](pass60_analysis.json) · [Attribution](attribution.md) · [Standard selected-endpoint summary](measured_pcqm-demol-droppath-100k-20261125-s42.json)
