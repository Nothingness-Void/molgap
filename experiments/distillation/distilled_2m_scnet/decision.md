# P8.17 Exact-2M Teacher Distillation Decision

## Decision

The 30%-teacher GPS7 student passed the internal gate but failed fixed external
evaluation. Reject both settings as global compression replacements. Neither
student changes the production model.

## Internal Exact-2M Test

| Model | HOMO MAE | LUMO MAE | Gap MAE | Average MAE |
|---|---:|---:|---:|---:|
| Fixed four-pass teacher | 0.09787 | 0.09537 | 0.12733 | 0.10685 |
| GPS7 student, 70% teacher | 0.09662 | 0.09321 | 0.12504 | 0.10496 |
| GPS7 student, 30% teacher | **0.09419** | **0.09154** | **0.12178** | **0.10251** |

The conservative student improves average MAE by `0.00245 eV` and Gap MAE by
`0.00325 eV` versus the 70%-teacher student. It improves average/Gap by
`0.00435/0.00555 eV` versus the teacher on this exact split.

The 70%-teacher run selected epoch 0 and then regressed. The 30%-teacher run
selected epoch 22 and completed all 30 epochs, showing that preserving more
B3LYP-label loss was necessary.

## Artifact Acceptance

SCNet jobs `703633` and `703653` completed with exit code 0. Independent job
`704402` recomputed hashes and accepted both model files, test predictions,
all 40 embedding parts per student, and each 997,445-row FP16 fusion prefix.
The acceptance record is `../remote/overnight_20260723_acceptance.json`.

## External Outcome

The student improved OOD-1000 and PCQM4Mv2 Gap, but significantly regressed
common overall and P8-hard. The external decision is
`../distilled_2m_external_eval/decision.md`. Do not train a new fusion head.
