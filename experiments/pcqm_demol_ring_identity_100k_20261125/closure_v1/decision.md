# DeMol-CIN direct ring return 100K decision

2026-11-26 JST. `ring_specific` closes `CLOSED`. `ring_mean` closes `NEGATIVE_UNDER_CONTRACT` under the [frozen protocol](../protocol.md).

| Arm | Best EMA MAE (meV) | Best pass | Pass-60 EMA MAE (meV) | Give-back (meV) |
|---|---:|---:|---:|---:|
| `ring_mean` | 135.414 | 31 | 139.992 | 4.577741 |
| `ring_specific` | 134.635 | 35 | 138.426 | 3.791004 |

The frozen comparison is `ring_specific` minus `ring_mean`. The aligned pass-60 estimate is **-1.566 meV**, paired 95% interval **[-2.539, -0.615] meV** over the same 50,000 internal development rows. The main EMA checkpoint rescores matched the remote pass-60 trace within the scorer's 0.0001 eV validation tolerance. Give-back uses the unrounded native trace. The training trace has 60 passes, 46,860 updates and 5,998,080 presentations per arm. The measured allocation ledger records 10.651 T4-device-hours across both arms, including allocated idle time. Queue time was unmeasured.

Ring-specific return improves the overall cohort by less than 3.0 meV. The predefined strata are preserved with the endpoint evidence.

[Endpoint rescoring, file pins and paired results](pass60_analysis.json) · [Attribution](attribution.md) · [Standard selected-endpoint summary](measured_pcqm-demol-ring-identity-100k-20261125-s42.json)
