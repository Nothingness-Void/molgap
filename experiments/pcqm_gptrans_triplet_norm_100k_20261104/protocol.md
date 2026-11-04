# Triplet return and pair-update normalization: 100K ordering test

Desktop-owned; user authorized Kaggle1 submission of two candidates on2026-11-04.
No reference retraining. Base desktop d0ea8b84; reference is the accepted
uniform_wd_multi_ema endpoint from pcqm_gptrans_norm_ema_100k_20261104.

## Evidence review and alternatives

RML cards/lookup/gaps --ref origin/molgap-server queried at2fae3b6d:197local
and190server-only IDs, local priority. No server evidence is imported.
Desktop PairNorm improves its matched100K parent by3.142991meV. Server third-pair
aggregation improves its parent by3.049777meV and leads59/60observations;
attention is not better. Later-cohort frozen aggregation gain1.766964meV has a
CI crossing zero. This justifies a bounded interaction test, not an assumed
sum of gains or500K nomination. Server true-bond routing regresses1.413165meV;
path/EMA combination, K1 sparse triplets, pair-memory routing, extra chemical
feedback and the desktop WD/EMA grid are not being retried.

Server sources (read at the pinned ref, not copied):
https://github.com/Nothingness-Void/molgap/blob/2fae3b6d/experiments/pcqm_gptrans_triplet_communication_100k/gpu/results/interpretation.md
https://github.com/Nothingness-Void/molgap/blob/2fae3b6d/experiments/pcqm_gptrans_triplet_portability/decision_terminal.md
https://github.com/Nothingness-Void/molgap/blob/2fae3b6d/experiments/pcqm_gptrans_triplet_routing_100k/gpu/degree_local_triplet_bond_route_ema999/results/decision.md

Hypothesis: normalization controls the useful scale of a new relation return;
it may instead erase useful amplitude, or the extra return may be redundant.
These mechanisms are unproven. This is not a replication of the old server
triplet placement, which modified pair state before GPA. Both new arms compute
the same T(pair) on incoming pair state, in parallel with the unchanged GPA.

## Frozen interventions

At blocks3/6/9/12 only, let D be the unchanged GPA pair update, P the incoming
pair state, N the accepted parameter-free pair-channel LayerNorm, and T the
third-pair-gated inward/outward aggregation (2heads,8channels/head).

- joint_norm: P'=P+N(D+T(P)).
- separate_norm: P'=P+N(D)+T(P).

Same trainable parameters, scratch seeds and exact parent tensors in both arms;
zero return projection starts both at the accepted parent. Preserve all valid
and virtual pairs, including[0,0], padding masks and unchanged node/readout
branches. No local mask, persistent third-order state, attention, geometry,
teacher, new feature/cache, auxiliary objective or auxiliary EMA is added.
The prospective question is update-order interaction, not automatic module stacking.

## Contract and reference

Fixed100K train[0,100000),50000selection-used development[100000,150000);
seed42,FP32/noTF32,physicalBS128/drop-last32,60passes/46860updates/
5998080presentations. AdamW LR0.001,WD0.05,clip1,warmup4/cosine to1e-6,
EMA0.999,minimum-development-EMA selection. Retain the accepted accelerated
foreach optimizer, exact OGB features/target transform and parent initialization.
The reference's unused auxiliary EMA0.9999 is removed for both candidates;
primary optimizer/EMA semantics remain the declared comparison subject.
Official validation/test_dev/test_challenge are forbidden.

Each candidate must bind the accepted retained desktop reference, including
checkpoint/predictions/trace/native runtime/roles/cost/target asset. Server's
foreach=false results motivate the hypothesis only, never the strict reference.
Missing compatibility blocks submission; do not retrain a baseline to fill it.

Each candidate's primary gate is delta<=-0.003eV and paired-row95%CI upper<0,
2000resamples,seed42. Report both arm results, including failures. The direct
A/B comparison is secondary, not evidence that either beats the parent.
Reused development-row intervals do not measure training seed uncertainty.
No automatic500K, full, retry or extra seed is authorized.

## Execution and closure

One Kaggle1 T4x2 notebook, one isolated candidate per GPU. Estimate4wall hours;
ceiling21600wall seconds/12allocatedT4-hours with180second cleanup reserve.
Keep epoch-only shared training logs with MAE/best/steps/exposure/LR/ETA.
Use native preflight for real optimizer-repeat/resume/memory qualification.
Use the existing CPU synthetic sandbox once for the new model boundary,
not a new test suite. No training or real-role inference occurs locally.

Reuse family training, atomic checkpoints, selected predictions, canonical
trace, source publication, submit_workflow, strict minimal retrieval, accept_run
and terminal RML closure. Require two distinct complete replay_pool entries
without exclusions. Preserve role/cost evidence, primary selected models,
aligned predictions, exact source, native proofs and final resumable state.
Download only the manifest-required minimum to D:. Luna owns remote checks.
No shared trainer/verifier modification is included in this experiment.
