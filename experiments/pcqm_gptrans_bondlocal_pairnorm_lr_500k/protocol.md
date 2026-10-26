# GPTrans BondLocal plus PairNorm: 500K learning-rate pair

Prospective desktop question, authorized by the user on 2026-10-27 for Kaggle1.
No full training, official evaluation, automatic retry or model adoption.

## Evidence and question

The accepted same-job 100K pair improved from 0.142861308 to 0.139718318 eV
by normalizing fresh pair updates; conditional gain 3.142991 meV and paired
row interval [-4.056945,-2.271359] meV. Its EMA-selected candidate peaked
at pass 40 then worsened 0.864 meV by pass 60. Retained degree-BondLocal 500K
peaked at 0.101951376 eV at pass 45 and rose to 0.102401979 at pass 60.
That older 500K candidate lacks a prelaunch strict reference and is not strict
replay-ready. Its score is descriptive context, not a qualified reference or
proof of the effect of PairNorm. Earlier standalone PairNorm and NoisyNodes+
PairNorm 500K were negative under their 3 meV material gates.

Hypothesis: lower peak learning rate reduces late development degradation on
the combined model at 500K. The curves do not identify learning rate as the
cause; model stochasticity, selection noise and other mechanisms remain viable.
The two arms are 500K-new variants of the accepted 100K combination: A uses
peak LR 1e-3, B uses 5e-4. Their same-run comparison isolates the specified
optimizer change, not PairNorm versus unnormalized BondLocal.

## Training and roles

Both: degree-scaled true-bond GPTrans256/pair32/local64, fresh pair-update
normalization, identical transported seed42 initialization, fixed 500K train
[0,500000), internal selection-used development [500000,550000), AdamW
weight decay .05, 4-pass warmup plus 60-pass cosine to 1e-6, EMA .999, best
development EMA, FP32/no TF32, batch128 drop-last, 60 passes, 234360 updates,
29998080 sample presentations, native T4x2, one arm per device. The 100K
development rows are inside 500K training and must never be treated as an
independent 500K evaluation. Official validation/test and other sealed roles
are excluded. Per-pass logging uses the shared epoch logger.

The original historical 500K run used a float64 target-hash encoding. This
new shared-family output stores float32 target tensors; its pinned hash comes
from the same retained ordered labels, not an altered dataset. TRAIN-only
normalization retains the historical GPTrans 500K values.

## Gate, cost and closure

Primary candidate-minus-reference B-minus-A MAE must be <= -0.003 eV and
paired-row bootstrap 95% CI upper bound < 0 on aligned 50000 development
rows (2000 resamples, seed42) to nominate B. Smaller numerical gains fail the
predeclared material gate; incomplete output is INCONCLUSIVE or STOP_FOR_COST.
Row bootstrap is not seed uncertainty. No early stopping, automatic retry,
full-scale training, official roles or model adoption. A is the new PairNorm
500K route and a same-run reference to B, not a rerun of old BondLocal.
Comparison of A against old BondLocal remains context-only unless a separately
validated historical reference package qualifies prospectively.

Budget: one Kaggle1 T4x2 allocation with a 42900-second entry ceiling, 180s
cleanup reserve and 23.833333 allocated T4-device-hour ceiling. Historical
T4x2 BondLocal 500K entry was 41606.29s / 23.115 T4-device-hours; changed
model and platform conditions mean completion in one allocation is not assured.
The existing periodic atomic checkpoint/chunk, package identity, actual run
receipt, minimal retained per-arm model/prediction/trace/resume/native proof,
role and measured allocation records are mandatory. Any budget stop preserves
truthful partial state without an automatic continuation.

Both arms require separate strict native/RML replay qualification. Scientific
completion, replay, optimizer comparison and adoption are separate decisions.
Use the existing experiment CLI, registered family training and Kaggle adapter;
retain only acceptance-required outputs on D:.
