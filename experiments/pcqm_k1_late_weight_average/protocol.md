# Frozen K1 late-weight averaging — 2026-10-08

Owner: desktop, dedicated branch from verified desktop8e1fb4da. The user
authorized testing and explicitly requested local execution without training;
A100 is reserved for training. Use local CPU only, deterministic FP32/noTF32,
four intra-op threads, worker ceiling600seconds. No optimizer or gradient.

## Evidence and question

K1 clean-BN selectedepoch49 MAE0.103658948eV is near the selected GPTrans live
MAE0.103826756eV, but above its EMA0.101887690eV. The latter's same-checkpoint
live/EMA difference is1.939meV, descriptive and not an isolated training effect.
Accepted K1 BN adaptation recovers1.245meV. Its raw epoch60 endpoint is worse
than selectedepoch49, while small clean-cohort probes did not identify a
uniform late-fit gain. No K1 per-step EMA state is retained in this accepted
500K training history. The separate100K EMA startup failures provide no score.

Question: can a fixed equal learned-parameter average of the accepted epoch49
and epoch60 states recover a useful part of the remaining error after identical
clean BN estimation? This cheapest local test probes late checkpoint averaging,
not true stepwise EMA. Failure does not reject EMA in general, and success
does not authorize EMA training or independent-role transfer.

## Frozen cases

Reuse the previously accepted native CPU selected-original and selected-clean
50K predictions, requiring exact hashes and first128 reconstruction. Strictly
load epoch49(selected) and epoch60(final) with the accepted source/factory,
3,658,817 parameters and equal target normalization. Average parameters only,
exactly0.5 each; raw averaged state uses epoch49 nonparameter buffers, with no
buffer averaging. Original checkpoint bytes and learned input states remain
unchanged. No ratio, checkpoint or calibration-size search.

| Newly executed state | BN state |
|---|---|
| epoch60 | saved original |
| epoch60 | fixed clean calibration |
| equal epoch49/60 learned parameters | saved epoch49 buffers |
| equal epoch49/60 learned parameters | fixed clean calibration |

Each clean calibration uses the existing recalibrated_batch_norm owner: exact
16,384 unique train members drawn by NumPyseed20261008, original draw order,
BS128, dropoutoff,128 cumulative BN updates after reset, no label objective.
Verify parameters/non-BN buffers unchanged during calibration and restore exact
original state/first128 inference afterward. Retain averaged parameters, BN
snapshots, predictions and checks. Reuse already materialized pure2D graphs;
do not decode the whole500K backing dataset again.

## Roles, comparison and decision

Training input:16,384 members of[0,500000), feature-only BN estimation. Loading
these retained graphs also reads their labels; labels are not optimized.
Evaluation: all50K historically selection-used internal-development members
[500000,550000), identical row order/targets. Official validation, test-dev,
test-challenge, common and OOD remain untouched. No geometry is constructed.

Primary contrast: selected-clean absolute error minus average-clean absolute
error. Existing paired_bootstrap_mean,1,000draws/seed20261008. Nominate this
specific averaging route only if point gain>=1meV and95% row lower bound>0.
Secondary descriptive controls: final-clean versus selected-clean, average raw
versus selected raw, and each new state's clean-versus-raw difference. A fixed
equal prediction blend of selected-clean/final-clean separates two-forward
output averaging from one-model parameter averaging; retain its predictions.
GPTrans EMA is contextual on aligned rows, with no architecture-causality claim.
No multiple-testing correction or independent generalization/seed-variance
claim; no scalar fitting, model selection or protected-role promotion.

Complete diagnostic closes NO_TRAIN regardless of the nomination outcome.
Record measured CPU wall/process time; accelerator/queue not applicable.
Timeout/identity mismatch remains incomplete, with no remote fallback or retry.
Reuse RML plan before execution and finalizer afterward, then rebuild/check.
Reviewed diagnostic evidence/reusable implementation uses BRANCHES' non-promotion
desktop route; model recommendation remains unchanged.
