# Decision — 2026-11-06

`full12` closes CLOSED; `full8_query4` closes NEGATIVE_UNDER_CONTRACT.
The candidate fails the frozen +1meV quality non-inferiority requirement.
No adoption, automatic successor or500K release follows. Runtime qualification
and strict replay readiness are separate outputs of the shared finalizer.

Kaggle3 job142861431/version1 completed all60 passes for both arms; frozen source
`b886a814ad4931a36a70a06c9e92ebbd71674700` and identities remain in STATUS.md.

| Arm | Selected development MAE, eV | Selected pass | Steps | Presentations |
|---|---:|---:|---:|---:|
| A full12 | 0.139440192079 | 42 | 46,860 | 5,998,080 |
| D full8_query4 | 0.140933614525 | 53 | 46,860 | 5,998,080 |

D minus A is +1.493422meV, with paired-row95% interval[+0.533633,+2.409157]meV
from2000 aligned-row bootstrap resamples. Both the point effect and upper bound
exceed the +1meV frozen margin. Hence the joint quality/cost gate fails even
without a mature-epoch timing estimate. These intervals condition on one seed
and the selection-used internal development role; they are not seed variance.

Measured process walls were9817.247282s (A) and9314.622664s (D), a5.12% reduction
over that process scope. This is not the protocol's median pass10–60 timing,
nor isolated inference latency or billed cost. Allocation accounting belongs
to the retained execution ledger and must not be added once per arm.

Four-arm cross-slot rankings are descriptive. The companion C-versus-B positive
result does not overturn this same-job comparison against the full12 incumbent.
See [attribution](analysis.md) and closure_v1 for measured facts and role audit.
