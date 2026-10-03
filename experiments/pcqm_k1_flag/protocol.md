# K1 FLAG bounded contract - 2026-10-02

Question: does supervised atom-embedding robustness improve clean PCQM4Mv2 Gap
prediction in the preserved K1/EdgeState single model?
User authority: detailed local RML analysis, one justified new experiment, Kaggle3
submission. No shutdown, monitor, server takeover or protected evaluation release.

## Fixed scientific intervention

Original atom192 / persistent directed bond64 / latent slot64 / active-slot1 /
nine local blocks / mixers3,6,9 / RWSE16 / mean pooling / scalar Gap head.
Stored/trainable parameters must remain3658817 (user2x ceiling7317634).
Only training uses FLAG at the categorical atom embedding output before RWSE:

1. Draw delta0 independently from uniform[-0.001,+0.001] for each atom/channel.
2. Evaluate normalized Gap L1 three times on the same physical minibatch.
   Divide each loss by3 and accumulate model gradients.
3. Between evaluations take delta<-delta+0.001*sign(grad_delta), detach and
   reset perturbation gradients. Model weights remain unchanged until the end.
4. Clip accumulated model gradient at1.0 and perform one original AdamW update.

No projection/clamping; maximum coordinate magnitude is0.003 across these three
evaluations. Each pass has the original dropout behavior. Geometry, bonds,
categorical identities and targets are unchanged. Perturbations/hook are ephemeral;
clean inference and saved model state have no extra parameters or perturbation.
Do not infer a label-preserving physical transformation from embedding distance.

## Roles, recipe and reference

Accepted fixed cache `nvoid912/pcqm4mv2-ogb-fixed-100k-v1`, manifest
`1b0e8fd579ab1cb86c02e833e7ad284b4af7582b059f912a77853fdccf3ede6d`.
Use only train[0,100000) and consumed development[100000,150000).
Discard retained geometry fields before the model; construct no3D.
No official-valid, test-dev or test-challenge read is permitted.

Seed42, frozen original initial tensor SHA
`8ef6d4ba1abdca04d8740a11ec0e04587358117b1e9f9534b4fc19d2b6caedbd`,
FP32/noTF32/deterministic, physical BS128/drop-last, original Python epoch order,
AdamW lr4e-4/wd1e-5, cosine40/eta1e-6, normalized Gap L1, no EMA, best clean
development live selection,40epochs and31240 optimizer updates.
Optimizer-batch presentations3998720 are unchanged. Gradient-bearing perturbed
loss-row evaluations11996160 are separately recorded; this is not an
equal-compute or equal-loss-exposure comparison. Train metrics average adversarial
dropout passes and cannot be compared as clean fixed-cohort train fit.

Reuse independently accepted original192/slot64 reference custody in
`experiments/pcqm_k1_slot_width96/reference_binding/`, MAE0.1412944608205557eV,
selected checkpoint4a7dab43..., predictions857fe314.... No baseline retraining.
Retain exact50K source/target hashes and literal role identities. The producer
runtime/software tuple is independently checked; missing equivalence prevents
strict causal/replay qualification and promotion, not artifact retention.

## Stages and stop rules

Before any model diagnostic, freeze a separate CPU prospective record and source,
reference, graph and row identity. First256 training rows only, CPU4threads,
300s ceiling; no development labels used by qualification. Require exact clean
eval identity after zero perturbation and after hook cleanup; finite nonzero
perturbation/model gradients; unchanged state storage; exact three-pass optimizer
accumulation checked against an independently spelled-out gradient calculation
with restored RNG; model/optimizer/RNG roundtrip; finite real pure2D output.
Sensitivity is descriptive, not a scientific promotion gate. On failure close
qualification NO_TRAIN and do not submit a training job.

Before formal training, remote T4 real-BS128 qualification must pass all identity,
repeatability, clean-eval, next-step resume and selected-state roundtrip checks.
Report synchronized FLAG/reference step ratio and peak memory separately. A
step ratio above4.0 stops before formal training (resource veto, not accuracy).
Expected assigned-device training window3-5T4h is an estimate; one kernel with
one formal candidate and two visible T4s, assigned device0. Idle-device/full
account/CPU/queue/bootstrap costs remain unknown unless measured. Kaggle wall
limit12h; no duration reduction/automatic restart changes the scientific recipe.

## Decision and durability

Complete40epochs before judging the candidate; no calibrated early-stop rule is
enabled. Reproduce aligned finite50K predictions with exacttargets and report
paired reference-minus-candidate gain and1000-replicate seed42 row bootstrap.
Nomination requires point gain>=3meV and positive lower95% row bound; seed
stochasticity remains unknown. Nomination does not release500K/full/official roles.
Strict reference/runtime/V5 transfer/cost gates remain separate from this endpoint.
Below gate: NEGATIVE_UNDER_CONTRACT, attribution then complete-history archive,
desktop canonical discovery, no successor. Infrastructure failure/unknown state
remains separate and requires exact-attempt reconciliation, not resubmission.

Freeze immutable source/config/input/initial identities, atomically checkpoint
model/AdamW/cosine/RNG and exact complete-epoch cursor, retain selected weights,
source-aligned predictions, contract, full trace, runtime certificates and native
cost; bound each in a retrievable manifest. Reuse existing family output and
platform adapters. CPU acceptance is a separate milestone, not formal training.

Comparison binding: the registered training_objective_comparison declares loss_identity and architecture_config_identity. The latter includes the training-only addon configuration; stored inference architecture is unchanged. Every other planned identity must match. Planned comparability does not certify future runtime equivalence or isolate adversarial direction from repeated dropout/loss evaluations.
