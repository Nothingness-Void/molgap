# archive-r02 PubChemQC Oracle Probe Decision

Primary gate: 10k label-blind random molecules from the independent 100k pool.
The other 10k descriptor-diverse molecules are diagnostic only.

| subset | n | fixed route | fixed precision | fixed recall | fixed Gap MAE | budget Oracle Gap MAE | unrestricted Oracle Gap MAE |
|---|---:|---:|---:|---:|---:|---:|---:|
| random | 9991 | 12.5% | 49.9% | 12.9% | 0.145693 | 0.137261 | 0.131211 |

Budget-matched Oracle adds **0.008432 eV** Gap improvement over fixed v4
(paired-bootstrap 95% CI `0.007916` to `0.008978`).

**Decision: GO.** Expand the independent pool and build Router train/validation/sealed splits.
