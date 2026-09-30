# Expert Oracle feasibility disposition, 2026-09-30

## Accepted diagnostic and disposition

NO_TRAIN for release of a new prediction-only router. The local diagnostic
is complete; its analysis implementation and evidence are adopted for research
reuse only. The model recommendation is unchanged. Await the separately owned
pretraining results and follow experiment_plan.md; no remote job is released.

| Method on original K1/GPTrans | MAE (eV) |
|---|---:|
| Fixed 50:50 | 0.100655621 |
| Cross-fitted global L1 weight | 0.100599195 |
| Prediction-only hard gate | 0.104561995 |
| Prediction-only soft gate | 0.100319860 |
| Label-informed discrete Oracle | 0.080244554 |
| Label-informed convex interpolation Oracle | 0.075036477 |

Hard gating worsens the global blend by 3.963 meV, paired 95% interval
3.621..4.308 meV; all five folds worsen. Winner AUC is 0.513674. Soft gating
improves by 0.279 meV, interval 0.194..0.366 meV; all five folds improve,
but the frozen 1 meV nomination criterion fails. Its gain is ~1.37% of the
20.355 meV discrete Oracle gap below the cross-fitted global blend. Statistical
detectability on consumed rows is not materiality or independent confirmation.
The published helper's 3 meV probability field is ancillary; it is not this
diagnostic's 1 meV rule. The retained three-model local-only fusion's historical
0.099902 score is also stronger than this two-model soft gate, under its own
exploratory analysis. No claim of a new best model is made.

## What the Oracle says

Five single-arm choices yield 0.059824820 eV; adding the original fixed blend
as a fallback yields 0.058878622. The weaker EdgeState is the best individual
prediction on 19.126% of rows within the five-arm Oracle. This supports residual
complementarity, not an identified structural competence. More choices lower
the Oracle even for noisy predictors; it is not a deployable score or evidence
that a gate can locate those rows.

With K1 as base, a perfect switch calling GPTrans on the most useful 5%, 10%,
and 20% of molecules yields 0.095873, 0.091218, and 0.085460 eV. The allocation
uses true labels; these are conditional-compute ceilings, not actual savings.
The executed prediction-space gate needs both predictions and is postdispatch.

21.518% of pair residuals have opposite signs. Fixed averaging beats both
individuals on 11.308% of all rows; those rows cannot be recovered by merely
choosing the less-wrong individual. This supports cancellation as one component
of fusion's advantage. Signed residual correlation remains 0.839531.

## Failure-mode attribution and limitations

Artifact and row/target identity passed. All five retained inputs match accepted
SHA pins, 50K ordered development rows and identical finite targets. No model
training/inference or protected-role access occurred. Geometry training replay
qualification remains unresolved; no causal geometry/promotion claim follows.

The hard-gate loss is supported as a failure of the tested prediction-only
winner classification. The small soft-gate gain supports limited weight
adaptation. Missing node/scaffold embeddings make richer routing unidentifiable;
this neither condemns all specialists nor proves sufficient exposure or a
physical feature deficiency. Model-level selection already consumed these rows;
five-fold gate cross-fitting does not remove that selection or measure seed
uncertainty. Bootstrap intervals condition on fitted OOF predictions.

Analysis cost: 5.029348 wall seconds and 5.000 process CPU seconds, excluding
imports, literature review and subsequent verification. Accelerator/queue cost
is not applicable. Prospective cost was a <=15-minute estimate, not measured.

An action-source scalar in the first planner input accidentally retained the
old example's e77ac57e commit while the planner correctly froze desktop state
f3e9c8fc. The original prospective snapshot is preserved. acceptance.json records
this transcription discrepancy and independently binds the actually executed
71976042 source and exact analyzer/helper bytes. prepare.py was fixed to bind
its caller-owned action source. This diagnostic is not qualified training replay.
