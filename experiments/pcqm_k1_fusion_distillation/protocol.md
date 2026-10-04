# K1 fixed-fusion teacher distillation — 100K

User authorization 2026-10-04: attempt the proposed single K1 distillation,
then confirmed Kaggle3,100K,two new arms. Desktop owns this question. No baseline
retraining, 500K/full advance, official-role access or default monitor.

## Evidence and hypothesis

The accepted frozen 50:50 dropout mean2/consistency2 teacher achieved
0.137891341eV on a separate50K internal-development cohort, improving over the
best constituent by2.811987meV with row-bootstrap95% bounds[2.400614,3.220851].
Its two-pass inference wall was29.644s vs13.535s for one constituent.
Owner: experiments/pcqm_k1_consistency_fusion_transfer/terminal_decision.md.
The original two training verdicts and replay exclusions remain unchanged.

Compressing an ensemble into a student is supported in general by Hinton et al.
[Distilling the Knowledge in a Neural Network](https://arxiv.org/abs/1503.02531).
Its classification results do not establish PCQM Gap regression efficacy.
This experiment tests whether the measured local teacher benefit is learnable
through regression targets from the existing training membership. Teacher
training-set memorization, loss bias, and teacher error transfer are explicit
alternatives; no claim of guaranteed distillation benefit.

## Scientific delta

One ordinary clean K1 student per arm,3,658,817 parameters. Both start from the
same retained seed42 initial state, not from teacher weights. Each uses one
dropout-enabled training forward. Loss is normalized-label L1 plus
lambda*MSE(student normalized output,stopgrad teacher normalized Gap).
Weak lambda=0.1; strong lambda=1.0. Values are prospective probes, not fitted
to labels. No temperature, pseudo-label-only replacement, architecture change
or teacher execution during student training/inference.

Teacher is fixed clean eval/live epoch37 of the accepted mean2 and consistency2
states,unfitted 50:50. Before any generation,publish the cache prerequisite
prospective trajectory. Generate only source_idx[0,100000), original training
membership, on local RTX5060 FP32/noTF32, batch128,600s wall cutoff.
Verify both checkpoint hashes and source-identical model definitions; exact
ascending row identity,finite outputs,pure2D fields. Cache contains only
source_idx and teacher prediction_eV; no labels or development predictions.
Publish a private hash-bound train-only cache dataset. Bind the payload,
manifest,teacher states and normalization identities into each training recipe.

## Unchanged training and retention

K1 V4 official-training prefix100K; selection development[100000,150000).
FP32/noTF32,physical BS128/drop-last,seed42,40epochs,31,240 optimizer steps,
3,998,720 sample presentations. AdamW4e-4,wd1e-5,clip1,cosine40 eta_min1e-6,
noEMA. Preserve exact row sampler and pinned initialization. Kaggle3 T4x2,
one isolated arm per device,all-arm preflight before either training arm.
Repeatability and resume probes must include the real cached teacher loss.
Budget estimate up to4 T4 device hours per arm; no automatic retry or successor.
Checkpoint each epoch with model/optimizer/scheduler/RNG/row cursor; selected
model,predictions,trace,cost,runtime certificate and completion manifest use
the existing k1-screen-v1 FamilyOutputSession protocol. Retain independently
retrievable per-arm artifacts for terminal RML; queue or submission is not
mechanical/scientific acceptance. Each arm needs its own terminal/RML record.

## Analysis and prospective gate

Historical clean K1 reference and the accepted consistency teacher constituent
remain saved-artifact references; runtime/role/recipe comparability is reviewed
at acceptance, and missing strict evidence remains missing. Compare both new
arms pairwise on identical development rows without pretending either is a
baseline. Also compare against retained consistency and the fixed teacher blend.
Primary compression nomination: at least1meV improvement over the stronger
retained constituent with positive paired-row95% lower bound, and no more than
1meV MAE loss relative to fixed teacher blend on the same old development rows.
Use1000 paired-row bootstrap draws,seed42; both strength candidates make these
exploratory unadjusted intervals. This is a nomination, not a V5 READY or full
promotion gate. Strict reference/runtime/cost/role gaps block such promotion.

Record label-only online train MAE separately from total objective. At closure,
analyze fit/exposure/teacher imitation/development errors where retained evidence
allows; online dropout train MAE vsclean dev MAE alone cannot diagnose overfit.
If a student is nominated, qualify its retained checkpoint on the separate
common development cohort under a new explicit role/prospective decision before
adoption. If both fail,close this direct-output teacher route without a blind
seed/schedule retry. Official validation/test roles stay sealed.
