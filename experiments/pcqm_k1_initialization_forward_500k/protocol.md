# K1 initialization under single and double forwards

Desktop-owned Track B question. User authorized two Kaggle1 slots and four
arms; no full-data training, protected roles, automatic retry or promotion.
Design date: 2026-10-26 JST. Parent source: desktop340800a1; subsequent reviewed
infrastructure commits must be recorded before source/prospective publication.

## Evidence and question

[Bounded attribution](../pcqm_k1_scale_attribution_20261025/analysis.md)
found that the strongest100K pair shared pretraining, so it did not isolate
pretraining benefit. Its teacher was omitted at500K. Recent MsgNorm/shell
variant gains were below their gates and did not establish original-K1 gains.
The negative500K tail improved sampled TRAIN fit while development regressed,
including after fixed calibration. Repeating tail decay or stacking modules
does not answer the unresolved initialization question.

Question: does the retained10-pass100K pretrained backbone improve500K
downstream generalization compared with the original random backbone, and is
the sign consistent under single-forward and mean-of-two-forward training?
Alternatives include no benefit, negative transfer, faster early optimization
without endpoint benefit, or benefit conditional on the forward recipe.

## Four arms

| Physical job | Reference | Candidate | Sole within-job change |
| --- | --- | --- | --- |
| single | scratch_single | pretrained_single | Frozen initialization |
| mean2 | scratch_mean2 | pretrained_mean2 | Frozen initialization |

The original seed42 head is byte-identical in scratch and pretrained states.
Pretrained backbone and BN buffers remain as retained; the intervention is the
entire retained initialization, not solely learned weights. This cannot isolate
pretraining objective, membership, horizon, weights versus buffers, or seed
variability. Each job trains two independent models on its two T4 devices.
No teacher, consistency penalty, architectural addon or extra pretraining is run.
Mean2 averages two stochastic supervised L1 losses; it is not prediction fusion.

Within each job all downstream settings match. Across jobs only the forward
recipe intentionally differs, but physical job/runtime realization also differs.
Cross-job contrasts and difference-of-differences are auxiliary, not declared
same-job STRICT_CAUSAL claims. Four models are diagnostic alternatives, not an
instruction to deploy a four-model ensemble.

## Shared downstream contract

500,000 fixed official-training members [0,500000); development50,000 members
[500000,550000), previously consumed. Pure2D OGB atom9/bond3/RWSE16; no geometry.
Official validation/test and unrelated roles remain sealed. Existing fixed500K
manifest630d30046d6cdc1f91fb169cd1eb4720bd5b352dc1ebeb641a11ead1ebae9751.
TRAIN-only frozen mean5.413434982299805/std1.17672860622406 eV, sample std;
retain the recovered TRAIN target hash, not development-only proof.

30 passes,117180 updates,14999040 sample presentations; batch128/drop-last,
seed42, torch-pass-randperm, FP32/noTF32, AdamW4e-4/weight-decay1e-5/clip1,
cosine-epoch30 to1e-6. Both forward recipes have equal optimizer/sample exposure;
mean2 has twice the stochastic forward views and greater compute/BN updates.
This is not compute-matched training.

EMA decay.999 and fixed TRAIN-only Clean-BN128 batches of128 in original first
16384 row order. Select best development Clean-EMA MAE over30passes. Record raw
Live development separately; never substitute a better post-hoc weight state.
This common output policy is a new shared condition, not a claim that historical
raw-Live scores are strict comparators. Calibration consumes TRAIN only and
retains its exact policy/order and state evidence. Use existing shared state
and selection lifecycle; no experiment-specific BN implementation.

## Decision and attribution

Primary per-job effect: scratch selected MAE minus pretrained selected MAE in eV.
Material nomination requires gain>=.001 eV and paired-row95% interval wholly
above0, using2000 resamples/seed42 after alignment and strict qualification.
Each interval is conditional on these fitted models and reused development,
not training-seed uncertainty or independent-test performance. Report both
comparisons, including negatives; neither permits automatic adoption/full run.
If both qualify positive, nominate the cheaper viable recipe for a separately
reviewed next stage. Opposing signs motivate interaction analysis; small effects
or failed qualification yield no promotion. Never use early curves to stop one
arm or silently change a horizon.

Retain complete per-pass TRAIN objective/development/learning-rate/exposure and
native-time curves; compare early and late matched-pass behavior, selected and
final endpoints, and raw versus Clean state where retained. Stochastic normalized
TRAIN loss is not directly a clean eV fit metric. If fixed-cohort inference is
needed later, declare it under its own diagnostic scope using retained states.
No claim of underfitting or capacity ceiling from an endpoint alone.

## Budget, closure and recovery

Each slot has at most32400 wall seconds including setup/qualification/evaluation
and180second cleanup reserve;18 allocated T4 device-hours per slot,36 total.
These are ceilings, not measured costs or a guarantee of completion. Record
actual allocated native T4/device/wall costs separately. Reused pretraining has
7810 updates/999680 presentations; historical device allocation remains unknown,
not zero. Downstream equal exposure does not mean equal total lifetime exposure.

Registered native all-arm qualification must pass before training. Keep atomic
resume state and bounded retained chunks, immutable source/config/data identity,
actual kernel/version receipts, selected model/aligned predictions/trace and
certificate-bound minimum native evidence. A budget pause is incomplete, not
replay-ready; preserve it for user-reviewed continuation, no automatic successor.
Retrieve only strict acceptance's minimal file set on D:. Each of four arms
must independently complete strict replay qualification and RML closure before
calling this experiment replay-ready. Local checks and RUNNING are not acceptance.

Remote status/startup checks belong to Luna. Reuse one owning monitor if explicitly
enabled; no server takeover or new conversation per tick. This request does not
authorize computer shutdown.
