# Clean-second terminal decision - 2026-10-10

**NEGATIVE_UNDER_CONTRACT; no adoption or scale-up.**

Exact Kaggle3 job `nvoid912/molgap-k1-clean-second-100k-s42-v1`,
ID137916849/version1, completed both fresh arms. The raw entry source matches
the frozen package. Both native qualification certificates and CPU-only
artifact inspections pass: 40 epochs, 31,240 optimizer steps and 3,998,720
optimizer-batch sample presentations each. Best live checkpoints are epoch37.
See [structured acceptance](result.json), [scheduler](scheduler_observation.json)
and [source observation](remote_source_observation.json).

| Gap on fixed50K selection-development | MAE (eV) |
|---|---:|
| Fresh mean2 reference | 0.140257285566 |
| Clean-second candidate | 0.141020240435 |

Gain = reference minus candidate = **-0.000762954869 eV**.
Paired row-bootstrap95% interval is **[-0.001677093695, +0.000113363703] eV**
(1,000 draws, seed42). It crosses zero and fails the prospectively frozen
gain>=0.003eV gate. This is failure to qualify, not proof of population harm.
The cohort was repeatedly used for selection; row uncertainty is not seed noise.

Measured two-T4 entry-window cost is **5.443604603 device-hours** including
installation, preflight and assigned idle time. Entire platform-release cost
is unknown; queue and pre-entry/post-entry tails are excluded. Native cost,
mechanical acceptance, strict comparison and replay qualification are separate.
The original producer's unknown platform-version field is not backfilled.

Preserve the original prospective records and terminal RML alongside this
decision. Complete rejected history follows the archive route in BRANCHES;
desktop receives canonical discovery evidence, not rejected model changes.
No automatic retry,500K/full training, protected evaluation or production change.
Read [attribution](attribution.md) before selecting another training-view module.
