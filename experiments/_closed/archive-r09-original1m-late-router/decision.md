# R09 Original-1M Late Router Decision

## Decision

Close the original-1M late soft-blend route. No candidate reached the
precommitted `0.001 eV` improvement in both average and Gap MAE on the
scaffold-disjoint validation selection partition, so the original test
partition remained locked.

## Validation Selection Result

| Candidate | Average MAE (eV) | Delta vs always-dual | Gap MAE (eV) | Delta vs always-dual |
|---|---:|---:|---:|---:|
| always-dual baseline | 0.0783510 | - | 0.0899683 | - |
| fixed target alpha | 0.0783266 | -0.0000244 | 0.0899517 | -0.0000166 |
| fixed Gap-bin alpha | 0.0783324 | -0.0000185 | 0.0899515 | -0.0000168 |
| three-seed HGB alpha | 0.0785424 | +0.0001914 | 0.0902239 | +0.0002556 |

The small fixed-weight changes are numerical ties, while the learned alpha
regresses. This does not justify another routing layer or external evaluation.

## Integrity

- Kaggle kernel: `nothingnessvoid/molgap-original-1m-late-router`, version 3.
- Split SHA256: `3703312b168589a66ef25c8323cd695fe2baa6b8317cb793d573841fe25a38b2`.
- Fit/selection partition: deterministic scaffold-disjoint split of the
  original validation rows.
- Pass rule: average and Gap each improve by at least `0.001 eV`.
- `selected=null`, `test_opened=false`; no test prediction artifact exists.
- No common, OOD, PCQM4Mv2, or sealed set was mounted for selection.

Raw outputs are retained under
`results/kaggle/evaluation/runs/molgap_original1m_late_router_v3/`.
