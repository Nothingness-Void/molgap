# One predictor versus an ensemble under total-system cost

Evidence synthesis and proposed experiment, 2026-10-28. Desktop owner:
`codex/exp/single-model-total-budget`, base `e6d514d5`.
The user subsequently authorized the local short execution profile below.
No scientific training run, protected-role access or remote submission is released.

## Authorized local short profile

Compare256/32/8 and384/48/12 serially on the RTX5060, with12 layers, local64,
pair-update norm, EMA.999, physical batch128, FP32/noTF32 and foreach AdamW.
Use only the first training shard of the existing100K cache; no development
inputs/labels and no geometry construction or coordinates passed to the model.
The fixed order is256,384,384,256, each with3 warmup and20 measured steps.
The synchronized window includes CPU collation/layout, H2D, forward/backward,
gradient clipping, optimizer and EMA. No quality comparison is performed.
Total GPU-allocation wall limit is600 seconds; stop on OOM or budget exhaustion
without changing batch/precision or stopping another process.
The [thin profile](profile_local.py) publishes a prospective before input
decoding/model construction, reuses the retained loader and shared update/EMA
owners, and saves source and raw results outside Git under
`D:/molgap-artifacts/local_profiles/single_model_total_budget/`.
This local result does not directly predict T4/A100 speed or500K accuracy.

### Measured local outcome

The authorized profile completed in20.85 seconds of measured allocation-window
wall time. All four windows used the same training-prefix batch sequence.

| Shape | Parameters | Per-window median step (ms) | Peak allocated (GiB) | FP32 parameter bytes (MiB) |
|---|---:|---|---:|---:|
| 256/32/8 | 5,871,201 | 93.38;75.99 | 1.031 | 22.40 |
| 384/48/12 | 12,478,993 | 102.27;101.92 | 1.537 | 47.60 |

The mean of per-window medians gives a1.206x time ratio. Baseline window drift
is material; cross-window ratios span1.092-1.346, not a confidence interval.
The first baseline may include cold CPU/cache/device effects. Do not present
the midpoint as a precise platform-independent multiplier or a throughput gain.
Memory covers live parameters, gradients, AdamW, EMA and training activations,
not other applications or deployment-only memory. No inference latency,
checkpoint overhead or full-epoch time was measured in this short scope.

Resource verdict: this candidate is worth considering for one bounded500K
accuracy study rather than rejecting for size alone. On this local window it
costs much less than executing the small model twice, but that is not a measured
full-training comparison against either ensemble. Accuracy and hardware/session
budgets still decide whether to launch; no500K job was launched here.

Window summaries: [profile_result.json](profile_result.json). The source archive
and identical result remain in the external artifact directory above. Per-step
timings and separate forward/backward/update timings were not retained.
The existing pair-normalization and topology suites passed22 tests; no new
test file or platform infrastructure was added.

## Objective and accounting

Find a single pure-2D Gap predictor that matches or exceeds the ensemble with no
higher total cost to train from scratch on the full dataset, submit and deploy.
Existing 500K weights are useful research assets, not free future full training.
Count every member's training, required pretraining/calibration, prediction and
checkpoint storage. Report exploratory search costs separately. Count shared
preprocessing once; count candidate-only features separately. Separate queue,
wall time, allocated device-hours, process CPU time and measured GPU busy time.
Do not add overlapping allocation and training-window records.

## Local performance and cost anchors

All scores here describe the same selection-used internal 50K development cohort,
not official validation or independent model selection.

| 500K system | Gap MAE (eV) | Predictors |
|---|---:|---:|
| BondLocal + pair-update norm + EMA, LR 1e-3 | 0.099987 | 1 |
| Two GPTrans, equal mean | 0.097525 | 2 |
| K1 plus both GPTrans, equal mean | 0.096301 | 3 |

[Fusion decision](../pcqm_k1_gptrans_pairnorm_fusion_500k/decision.md) and
[inputs](../pcqm_k1_gptrans_pairnorm_fusion_500k/analysis.json) own those scores.
The approximate parameter totals are 5.87M, 11.74M and 15.40M respectively;
parameter counts and predictor counts are not measured latency or FLOPs.

The two GPTrans training windows are 10.262424 and 10.414019 hours. Their whole
two-T4 allocation totals 21.080753 T4-device-hours; do not add this to the windows.
See [native accounting](../pcqm_gptrans_bondlocal_pairnorm_lr_500k/terminal_acceptance/qualification.md).
The actual ensemble K1 used Stage10 pretraining with a reset head and a complete
60-pass T4 training/evaluation window of 21.606289 hours; the selected endpoint
was epoch49, followed by clean-BN calibration. The three model windows sum to
42.282732 hours, not a complete system bill or parallel elapsed time. Required
pretraining and other missing costs remain unknown, not zero.
See [K1 source](../pcqm_k1_500k_bottleneck_diagnostic/terminal_decision.md) and
[calibration](../pcqm_k1_500k_bn_calibration/terminal_decision.md).
Future full-dataset costs for these exact systems have not been measured.

## What past experiments rule out

Old GPTrans node352 kept pair32 and eight heads (44 channels/head), unlike an
official proportional enlargement. Under its 100K contract, MAE worsened from
0.144233 to0.147102; terminal online training fit was nearly unchanged. FFN2
also failed to provide a reliable improvement. Neither included the present
BondLocal plus norm combination. See server commit `f426350a`,
`experiments/pcqm_gptrans_capacity_nodes_100k/gpu/results/interpretation.md`.
This weakens blind node-only widening; it does not compare a strong single
predictor against an ensemble at matched total cost.

Do not reopen [output distillation](../pcqm_k1_fusion_distillation/terminal_acceptance/decision.md)
or [prediction routing](../pcqm_pairnorm_prediction_router_500k/decision.md).
Server path-mean and first/last-bond inputs have already been explored; do not
present generic path encoding as a wholly new route. Its ordered official
variant is different, but expands feature/cache scope and is not shortlisted.

## Two complementary candidates

**Proportional GPTrans-S-like core with our retained local/norm/EMA recipe:**
12 layers, node384, pair48, 12 heads, 32 channels/head, FFN ratio1.
Retain BondLocal and normalize fresh pair updates, not accumulated pair state.
This changes the capacity package, not just node width; its result would not
isolate any individual dimension. This exact combination was not located in
the reviewed desktop/server RML. Remote training cost and predictive accuracy
remain unmeasured; local execution measurements are recorded above.

The [official T/S configs](https://github.com/czczup/GPTrans/tree/main/configs/pcqm4mv2)
use 256/32/8 versus384/48/12, with6.6M versus13.6M parameters and official-valid
MAE0.0833 versus0.0823. Those parameter counts are not our local variant's count.
The local input encoders differ from the official implementation.

The user requested a second candidate for a two-arm Kaggle study. Proposed arms:

| Arm | Layers | Node / pair / heads | Parameters | Local speed evidence |
|---|---:|---|---:|---|
| A, proportional width | 12 | 384 / 48 / 12 | 12,478,993 measured | Profile above |
| B, narrow depth | 16 | 256 / 32 / 8 | 7,700,481 measured | Local short profile passed |

Both retain BondLocal(local64), fresh pair-update normalization, EMA.999 and
32 channels/head. For B, the present module formulas give457,320 parameters per
block and383,361 outside blocks:16*457,320+383,361=7,700,481; local construction
confirmed this count. Do not call the profiled12-layer256 model arm B.

Arm B's separately authorized [local result](profile_b_result.json) used two
3-warmup/20-step windows: median109.40/93.35ms, peak allocated1.343GiB, FP32
batch128 with the same local/norm/EMA execution. This establishes local resource
feasibility only, not T4 speed or accuracy. Its source snapshot is retained under
`D:/molgap-artifacts/local_profiles/single_model_total_budget/candidate_b/`.

This is a precision/cost choice between two single predictors, not a pure depth
ablation, equal-parameter study or equal-compute claim. A deeper384 model was
considered but not selected: it would concentrate both arms on the costly width
direction rather than test a lower-parameter alternative. Narrow depth16 with
the present combination was not located in the targeted local/server RML review;
that does not establish novelty beyond the reviewed records. It is distinct from
node352 and pair-residual scaling studies. Changing depth also redistributes the
existing linear DropPath schedule; it is part of the architecture package.

Proposed common training controls: fixed500K train and the same permitted50K
selection-used internal development rows, seed42, physical batch128, FP32/noTF32,
AdamW peak LR1e-3 and WD.05, four-pass warmup/cosine,60 passes and234,360 updates,
EMA.999 with the same selection rule. Initialize each shape from scratch and
freeze its actual tensors; same seed does not mean identical initialization.
Reuse existing frozen prediction anchors; no baseline retraining is implied.
Before submission, separately freeze the actual hardware/account, per-arm and
total cost caps, evaluation/selection details and retry policy. This paragraph
was a design proposal, not release authority. The later user authorization is
Kaggle1, two arms,60 passes each, checkpoint continuation permitted, and at most
30 T4-device-hours across both arms and all attempts. The observed account quota
was15.454843 platform GPU-hours on2026-10-28; quota units are not assumed equal
to device-hours. Proposed first segment is7h wall/14 T4-device-hours including
setup and cleanup, below both ceilings. Subsequent segments require actual-cost
reconciliation and remaining budget; they are not automatically submitted.

The [GPTrans paper](https://arxiv.org/html/2305.11424) reports Small at5.5min/epoch
on8 A100s, half precision, batch1024. At300 epochs, constant-speed extrapolation
is27.5 wall hours /220 A100-device-hours, not a measured complete training bill.
Do not copy that full recipe as an affordable default.
[GPS-deep](https://github.com/rampasek/GraphGPS) is a useful cost anchor:
13.8M,16x256, approximately60h on one A100, official-valid0.0852. Different
splits and training schedules preclude treating these public results as a
controlled cost comparison or as predictions of our500K scores.

## Smallest proposed discriminator

1. After explicit resource authority and prospective publication, construct and
   profile each proposed candidate on allowed training inputs. Measure parameter
   count, peak memory, end-to-end throughput, checkpoint/resume overhead and
   single-model inference. No protected or official labels are needed.
2. Reject deployment-infeasible or over-budget configurations before a500K run.
   Select one hardware class for the comparison; do not convert A100 and T4
   hours by an assumed speed factor. A per-session limit and a total budget are
   separate, explicit configuration fields, with no automatic continuation.
3. If affordable, propose one500K realization per candidate (two arms total) on
   the same permitted fixed data/targets. Freeze the full learning schedule,
   selection rule and total measured-cost cap before training. The fixed60-pass
   recipe is the initial comparability anchor; do not change it mid-run to fit a
   cap. An unfinished capped run is STOP_FOR_COST, not a complete negative.
4. Reuse aligned retained small-model/fusion predictions as performance anchors;
   do not retrain them merely for bookkeeping. Runtime differences must remain
   explicit. A strict causal training claim requires separately qualified
   comparisons; a system-performance study is not norm attribution.
5. Primary ambition: reach0.096301 with no greater total-system cost. Secondary
   ambition: reach0.097525 while costing less than the two-model system. These
   are proposal targets, not amended promotion gates or independent-test scores.
   Freeze exact cost and any noninferiority margins only after profiling and
   user budget approval. Include measured inference and stored-weight size.

No claim about full-data ranking follows from500K alone. A favorable500K system
comparison would justify separately costing a full-data run, not launching it.
No existing V5, material threshold or registry recommendation is changed.

## Reuse boundary

Use the existing GPTrans family trainer, state/session, data loader, logging,
receipt and acceptance owners. The core accepts dimensions, but current local
and norm factories/catalog modes fix256/32/8; proportional expansion needs a
small reviewed constructor/catalog extension, not a replacement trainer.
See [core](../../src/molgap/gptrans.py),
[local block](../../src/molgap/gptrans_bond_local.py),
[norm variants](../../src/molgap/gptrans_variants.py),
[catalog](../../src/molgap/experiment_catalog.py) and
[quickstart](../../docs/operations/EXPERIMENT_QUICKSTART.md).
Platform skills own actual submission. This research brief creates no fake
trajectory, native measurement, execution authority or RML terminal outcome.
