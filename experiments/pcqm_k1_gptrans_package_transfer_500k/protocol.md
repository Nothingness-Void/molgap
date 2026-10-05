# Frozen teacher-free package transfer contract — 2026-10-05

Authority: the desktop user approved this two-arm500K60-pass execution after
reviewing [the proposal and evidence](plan.md). Implementation, source freeze,
qualification, submission and acceptance remain on the dedicated desktop-owned
`codex/exp/k1-gptrans-500k-package` branch. No teacher is authorized or consumed.

## Frozen scientific execution

| Arm | Native composition and initialization | Optimization and selection |
|---|---|---|
| `k1_pretrained_consistency` | K1-v4 atom192/edge64/slot64/layers9; retained100K10-pass backbone with original seed42 Gap head; 3,658,817 parameters | AdamW4e-4/wd1e-5/clip1; cosine60 to1e-6; two dropout label-L1 forwards plus0.1 disagreement MSE; train500K mean/unbiased std; best clean live development |
| `gptrans_g1_bond_local_ema999` | Accepted G1 random initialization; node256/pair32/layers12/heads8/FFN1; degree0.0897; real-bond local64 with zero output projection; 5,871,201 parameters | AdamW1e-3/wd0.05/clip1; warmup4/cosine60 to1e-6; normalized labelL1; pinned native100K target transform; EMA0.999 once per optimizer step; best EMA development with live diagnostic |

The executable values and exact initialization file/tensor hashes belong to
the owning `pcqm_composed_500k.scientific_contract` and generated per-arm
`training_recipe_*.json`. The canonical Spec pins their bytes. No override,
baseline retraining, geometry construction, teacher cache or fusion weight fit.
Both arms use seed42, FP32/noTF32, physicalBS128, no accumulation and
epoch randperm(seed42+epoch), dropping32rows per pass. Sixty complete passes
require234,360optimizer updates and29,998,080presentations. Resume restores the
same schedule and completed-epoch cursor.

## Data and protected roles

Use only accepted fixed500K dataset
`nothingnessvoid/pcqm4mv2-ogb-fixed-500k-scnet-v1`, manifest SHA256
`630d30046d6cdc1f91fb169cd1eb4720bd5b352dc1ebeb641a11ead1ebae9751`.
Train is source_idx[0,500000); internal development is[500000,550000), already
used for development/selection. Pure2D strips retained geometry. GPU
qualification uses training batches only. Official validation, test-dev and
challenge remain sealed. [Role declaration](role_plan.json) owns usages.

## Evidence review and comparison limits

The verified local canonical evidence is
`pcqm-matched-500k-v4-three-arm`, trajectory`TB-matched-500k-v4-three-arm`,
under [matched V4 evidence](../pcqm_500k_v4_evidence/v5_evidence.json).
It supplies contextual60-pass references and is not silently retrained.
External immutable sources are the desktop K1 teacher-free accepted decision
at [51bc61fc](https://github.com/Nothingness-Void/molgap/blob/51bc61fc/experiments/pcqm_k1_pretrained_consistency_teacher/terminal_acceptance/decision.md)
and the server G1/local/EMA interpretation at
[f426350a](https://github.com/Nothingness-Void/molgap/blob/f426350a/experiments/pcqm_gptrans_capacity_relations_100k/gpu/results/interpretation.md).
These external commit references are not fabricated local RML evidence IDs.
Their retained negative/subthreshold and qualification limits are preserved in
[the evidence review](plan.md). Existing matched500K and geometry endpoints
remain contextual unless strict source/recipe/runtime/artifact comparison passes.
Two different native recipes do not constitute a single-intervention causal Replay.

## Acceptance and decision

Accept each arm independently using exact source/Spec/data/job/version, runtime
certificate, complete60-pass trace and exposure, objective activity, aligned
finite50K prediction artifacts, selected state, atomic resumable optimizer/RNG/
sampler/EMA state, roles and measured native costs. Mechanical completeness
does not assert scientific comparison qualification or adoption.
The primary ensemble endpoint is the fixed arithmetic50:50 mean of the two
aligned selected prediction vectors. Complementarity nomination requires gain
at least1meV over the stronger component and paired-row95% lower bound>0;
1,000bootstrap draws/seed42. Component nomination requires at least3meV versus
its applicable qualified reference and favorable paired-row95% bounds. Missing
references/qualification remain pending; row bootstrap is not seed variance.
Finalize two prospective trajectories separately and record failure attribution
before another unrelated module. No automatic full training, official evaluation,
successor question or model promotion follows any endpoint.

## Cost, qualification and bounded continuation

Use explicit Kaggle1 T4x2 with one independent arm/device. The shared source,
frozen initial state, accepted graph/cache and native recipe are checked on CPU;
both isolated training-only T4 preflights must pass before either training arm
starts. A failure retains logs, truthful pair state and partial cost observations.
[Budget](budget.json) owns the48allocatedT4hour training ceiling, with observed
pair idle capacity included. Qualification costs are measured separately.
The18.69/18.80T4hour estimates are rough historical scaling, not measured new
training cost or a one-session guarantee. CPU, queue, wall and T4 units remain
separate, and missing history remains unknown.

Each invocation has a32,400second bound from bootstrap entry, subtracting
install/CPU/preflight elapsed time from worker training windows. The existing
trainer stops at a complete-epoch boundary using observed next-epoch estimates,
then publishes stage manifests, best/last, trace, optimizer/RNG/sampler and EMA
state under independently retrievable Kaggle outputs. An unfinished stage is
not a scientific terminal result. Continue only this same frozen question from
retrieved, hash-verified private checkpoint inputs after exact run reconciliation
and cumulative budget review. Parent/skill owns continuation preparation; no
automatic monitoring, retry, server takeover or platform submission is provided.
