# Proposed two-arm package-transfer protocol — 2026-10-05

## Question and selection

Do the selected teacher-free K1 composition and recent GPTrans composition
transfer to 500K unique training molecules with a full 60-pass budget, and
do their prediction errors support fixed equal fusion?
This tests complete recipes and transfer, not isolated module causality.
It does not establish these packages as globally best or production-adopted.

| Arm | Composition | Preserved scientific mechanism |
|---|---|---|
| `k1_pretrained_consistency` | K1-v4, atom192/edge64/slot64/9 layers, 3,658,817 inference parameters | Exact retained 100K-stage 10-pass backbone; original seed42 Gap head restored; two independent dropout forwards; averaged label L1 + 0.1 disagreement MSE; no teacher |
| `gptrans_g1_bond_local_ema999` | G1 GPTrans, node256/pair32/12 layers/8 heads/FFN ratio1; degree scale0.0897; real-bond local64; 5,871,201 parameters | Accepted G1 input/init behavior, interior real-directed-bond local stream with zero-output initialization, EMA0.999 updated once per optimizer step |

There is no accepted500K-trained fusion teacher. Evaluating100K-trained teachers
on500K rows would generate new predictions, not establish a500K-trained teacher
or its quality. Following the user's correction, this pair has no distillation
loss, teacher-cache preparation or teacher inference. Do not use downstream-
selected student Gap weights as initialization. G1 uses its owning random
initialization, not K1 pretraining.
No new module, geometry, width expansion, path encoding, precision change or
new fusion-weight search is part of this question.

## Evidence and limitations

- Desktop K1 owner commit [51bc61fc](https://github.com/Nothingness-Void/molgap/blob/51bc61fc/experiments/pcqm_k1_pretrained_consistency_teacher/terminal_acceptance/decision.md): the selected teacher-free arm completed40epochs and scores0.138265848eV, with independent complete Replay acceptance. Its teacher sibling scores0.136781212eV, but that sibling is not the proposed arm. The narrow100K teacher gate passes while compression fails; it does not qualify an absent500K teacher. Pretraining itself and its interaction are not isolated, so teacher-free transfer is a new scale question rather than a proven gain claim.
- Server authority [f426350a](https://github.com/Nothingness-Void/molgap/blob/f426350a/experiments/pcqm_gptrans_capacity_relations_100k/gpu/results/interpretation.md): local-bond G1+EMA999 scores0.1423582275eV versus G1+EMA9990.1442326291eV. Gain1.8744017meV with favorable paired-row bounds, below the frozen3meV gate. Preserve its INCONCLUSIVE/no-promotion outcome. The exact local comparison is strict and complete Replay, not a 500K architecture qualification.
- The server's genuine500K study uses one unchanged G1 live model with two EMA filters. Fast EMA0.1152574413eV versus slow0.1247831724eV supports the correction direction, but only46,860 updates/5,998,080 presentations (~12passes). It is PAIRED_ENDPOINT, not causal Replay or60passes. It contains no local-bond addon.
- [Matched500K V4](../pcqm_500k_v4_evidence/final_decision.md): K1 reference0.104860eV, GPTrans reference0.106868eV,60passes. Reuse artifacts; no bookkeeping baseline retraining. Their optimizer/initialization/selection/runtime differences from these compositions must remain explicit.
- [Historical geometry fusion](../pcqm_geometry_transfer_500k/decision.md): fixed equal blend0.0995931532eV, OOF blend0.0995458298eV. This remains important accepted contextual evidence, but absent matching V4 reference/runtime qualification forbids a global cross-contract ranking. This plan is pure2D, not a replacement geometry test.
- [K1 retained fusion qualification](../pcqm_k1_consistency_fusion_transfer/terminal_decision.md): equal fusion improves2.811987meV over its stronger constituent on source_idx500000:550000. This supports measuring complementarity, not predicting the new packages' fusion gain.

The new question is enlargement of these exact composed routes under a genuine
60-pass budget. Old negative/subthreshold decisions are not relabeled. Neither
100K-to500K MAE subtraction nor single-seed row bootstrap measures seed variance.

## Data and recipe proposal

Reuse the accepted fixed500K OGB/RWSE16 asset and its exact manifest:
train source_idx[0,500000), internal development[500000,550000). The latter
already has development/selection history; it is not a sealed test.
Pure2D strips geometry. Official validation/test-dev/challenge remain sealed.

Both arms: seed42, deterministic FP32/no TF32, physical batch128, no gradient
accumulation, epoch randperm seed42+epoch, drop_last32,60complete epochs,
3,906updates/epoch,234,360updates and29,998,080presentations. Count completed
epoch observations separately from checkpoints. Resume never restarts schedule.

A preserves AdamW lr4e-4/wd1e-5/clip1, no EMA, live clean selection; extend its
cosine horizon to60 with minlr1e-6. B preserves its native AdamW lr0.001/wd0.05,
warmup4/cosine60/minlr1e-6, dropout/drop-path0.1 and best EMA development
selection; retain live development predictions as a diagnostic. Preserve each
family's owning target-transform semantics: A train-derived statistics over
the new500K training cohort; B the pinned100K training-subset transform used
by G1. Bind actual assets and digests before release. These recipe differences
prevent an architecture-only causal comparison between A and B.

A reads actual500K training labels only. No teacher cache or new inference is
required. The retained pretraining backbone remains a100K-stage artifact, not
new500K pretraining. Its historical costs and qualification gaps stay separate;
unknown is not zero. Both new500K outputs can subsequently be evaluated as a
fusion candidate, not predeclared as an accepted teacher for another student.

## Endpoints and proposed decision gates

1. Accept each arm independently: exact physical job/version/source/Spec/cache,
   runtime certificate,60epoch trace, exact exposure/LR, objective components,
   selected state, finite aligned50K predictions, atomic optimizer/scheduler/
   RNG/sampler and EMA resume state where applicable, roles and native costs.
2. Report each selected MAE and exposure-indexed learning curve. Reused matched
   historical500K comparisons remain contextual until strict qualification
   actually passes. A proposed material nomination requires at least3meV over
   its applicable qualified reference with favorable paired-row95% bounds;
   no missing comparator is retrained or silently substituted.
3. Primary ensemble endpoint: arithmetic50:50 mean of retained aligned A/B
   predictions. Nominate complementarity if gain>=1meV over the better component
   and paired-row95% lower bound>0. Use1000bootstrap draws/seed42; no weight fit.
   This requires no third training arm. Development-selected checkpoints limit
   generalization claims even though the mixture weight is fixed.
4. Preserve two prospective and two independently finalized RML trajectories.
   Mechanical completeness and scientific comparison qualification are
   separate. Different families/objectives/optimizers/EMA/init cannot be forced
   into the existing single-intervention strict Replay world. Do not promise
   two causal replay-ready entries without applicable qualified references.
   If missing, record transfer/context exclusions and the exact missing
   discriminator using existing V5/RML owners; do not weaken their validators.
5. Complete attribution before another module: clean development curves,
   selected/terminal epochs, objective activity, train-metric timing, paired
   row residuals/cohorts, cost and remaining causal alternatives. A terminal
   endpoint alone cannot establish underfitting or exposure shortage.

## Execution budget and release prerequisites

Proposed platform: Kaggle1, explicit T4x2, one independent arm/device. Actual
account, quota and allowed session duration must be reconciled before release.
No default monitor or server job takeover. Server code/evidence is reviewed
input to desktop custody, not a wholesale branch merge.

A's measured100K40epoch training window is8,966.101allocated T4seconds;
linear presentation scaling to this endpoint is ~18.69T4hours. This is a rough
estimate: development/checkpoint time does not scale identically. B's notebook
half-allocation (~3.7585T4hours at5,998,080presentations) gives ~18.80T4hours
under the same rough5x assumption. These are not measured500K60pass costs or
wall-time guarantees. Reserve a proposed48allocated T4hour training ceiling
including paired idle capacity; measure qualification costs separately.
Do not claim the complete trial fits one overnight session.

Reuse bounded resumable invocation semantics: each invocation attempts all
remaining epochs, stops before the verified session boundary using observed
next-epoch estimates, and publishes exact best/last/trace/state chunks to durable
outputs. Continue the same question from an independently verified private
checkpoint input, with no source/recipe change or repeated completed epochs.
Do not pre-split into tiny arbitrary stages or rely on transient worker files.

Reusable owners: `pcqm_500k_v4_evidence.run` and scale cache/loader for60pass
sampling, deterministic recovery and bounded continuation; family model/loss/
EMA owners and the accepted K1 pretrained-initialization mapping; existing workflow,
source/release/receipt, Kaggle paired runtime/adapter, family acceptance,
`comparison_readiness` and RML terminal closure. The legacy500K loop does not
already implement these two compositions; current K1 screen constants and
server EMA rungs are100K/equal-update specific. Extend the owning family
adapter for explicit scale configuration, not an experiment-local copied loop.

Before any new execution: publish two prospective trajectories; review/import
only the required server and K1 reusable changes with provenance; bind actual
500K graph/cache/init/transform assets; register compatible scale execution and
acceptance; run one focused final regression batch and release validation.
Then actual training-only T4 fixtures qualify both arms behind the all-arm
barrier. A failed release/qualification stops and retains truthful evidence.

This document is a reviewable proposal, not a frozen executable contract or
submission receipt. No new training, inference or remote publication occurred
while writing it. Adoption/full-scale/official evaluation is a later decision.
