# Pretraining pair terminal-attempt review, 2026-09-30

Both workers completed their frozen source release3 on Kaggle1. Immutable
source/archive bytes, finite aligned 50K predictions, selected-model hashes,
pretraining checkpoints, downstream checkpoints and epoch histories were checked.
K1 completed 10+40 passes (7,810+31,240 optimizer updates); GPTrans completed
10+60 passes (7,810+46,860 updates). No new training or inference was performed
by this acceptance. Official validation, test-dev and challenge remained sealed.

| Arm | Accepted-artifact MAE (eV) | Historical reported reference | Point gain (meV) |
|---|---:|---:|---:|
| K1 | 0.141176449 | 0.141373634 | 0.197185 |
| GPTrans Noisy Nodes + PairNorm | 0.146358246 | 0.147245049 | 0.886804 |

Neither point estimate reaches the frozen 3 meV nomination requirement.
Strict same-family comparison is not accepted: historical reference prediction,
trace and readiness qualification remains incomplete. The terminal attempts
are therefore recorded INCONCLUSIVE, with no scale-up or replay-ready claim.
The scientific question remains pending artifact reconciliation, not a new run.

K1 raw metadata retains inherited Kaggle2/account and legacy run labels.
The actual launch receipt, pulled kernel, mounted dataset and native observation
identify Kaggle1/nothingnessvoid. Raw outputs are preserved unchanged; this
explicit discrepancy prevents silently treating the legacy comparator identity
as the actual new attempt. Checkpoint source/archive pins match release3.

The exploratory fixed 50:50 blend has MAE 0.134096619 eV, a 7.079830 meV gain
over pretrained K1, paired-row 95% CI [6.499953, 7.720366] meV. This uses the
already consumed development role, a single seed and no fitted blend weights.
It is not compared with a verified unpretrained blend and is not promotion
evidence for pretraining. Row bootstrap does not measure training stochasticity.

K1 selected downstream pass38 and did not improve by pass40. GPTrans selected
pass60 and its EMA development metric was still improving. Continued GPTrans
adaptation is compatible with that trace but insufficient exposure is unproven.
Online normalized live training losses cannot be compared directly with eV
development MAE (EMA for GPTrans). No raw fixed-cohort comparator was retained;
underfitting, overfitting and the causal mechanism remain insufficient_evidence.
The frozen code constructs EMA after pretraining, so initialization before
pretraining is not an observed EMA contamination fault.

Recorded process wall time is 6,765.609 seconds for K1 and 10,331.889 for
GPTrans, including stage/preflight overhead. These are assigned-arm process
timings, not GPU busy-time or scheduler billing. RML records measured wall hours;
device, CPU and queue accounting remain measurement_missing. The prior failed
attempt's measured process timings are retained separately. Pretraining and
downstream raw histories remain preserved; canonical downstream traces retain
missing optimizer/sample axes as null where absent in raw GPTrans epoch rows.

The cheapest remaining discriminator is verified historical same-row prediction
and trace retrieval plus strict V5 qualification. No scratch retraining, longer
schedule, new specialist or protected-role use is released by this decision.
