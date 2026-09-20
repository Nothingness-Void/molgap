# GPTrans propagation-flow decision

Kaggle1 kernel `nothingnessvoid/molgap-gptrans-flow-ablation-s42` version 1
completed both isolated 60-epoch arms. No-inference acceptance verified the
immutable reference comparison, source and archive identity, fixed 100K/50K
row order, exact seed-42 initialization, 5,246,817 parameters, strict FP32,
physical BS128, 46,860 optimizer steps, 5,998,080 sample presentations,
recoverable checkpoints, finite aligned predictions, and all artifact hashes.

| Arm | Development MAE | Difference vs GPTrans-T | Paired 95% interval |
|---|---:|---:|---:|
| Frozen GPTrans-T | `0.1566272043 eV` | reference | - |
| No direct pair-to-node | `0.1592841231 eV` | `+0.0026569188 eV` | `[+0.0015992863, +0.0037041772]` |
| No cross-layer pair recurrence | `0.1581743161 eV` | `+0.0015471118 eV` | `[+0.0005521753, +0.0025346946]` |

Both intervals are entirely unfavorable. Direct pair-to-node readback and
cross-layer pair recurrence are therefore independently useful under the fixed
100K contract. GPTrans-T's weak 100K result is not explained by either path
being redundant or destructive.

The outcome is `NEGATIVE_UNDER_CONTRACT`. Deletion-style propagation
simplification is closed without another seed, scale bridge, full training, or
protected-role access. A future GPTrans question must preserve both paths and
test a distinct mechanism such as conditional allocation or relation capacity;
this experiment does not authorize one automatically.

The T4x2 job used approximately `2.9182` wall hours and `5.8364` T4 device-
hours. No production model changed.
