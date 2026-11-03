# GPTrans normalization-decay and multi-timescale EMA500K

Prospective design,2026-11-03 JST. Desktop owns the question. User authorized a
Kaggle1 two-arm submission reusing local infrastructure and no separate baseline
training. No execution yet: registered runtime capabilities are pending.

## Evidence

Prior WD0.10 versus0.01 regressed0.604/1.131meV at initializations43/44;
these results do not isolate normalization parameter decay. Current optimizer
uses one decay group over model.parameters(). Historical same-recipe init noise
spans1.332meV. Same-recipe prediction ensembles improve development MAE; this
is not evidence that parameter averaging has the same effect.

Read ../pcqm_gptrans_pairnorm_wd_500k_s43/analysis_20261103.md,
../pcqm_gptrans_pairnorm_wd_500k_s44/decision.md, and
../pcqm_gptrans_pairnorm_init_noise_500k/decision.md.
Paper anchors: https://arxiv.org/html/2405.13698v3 (normalization parameters
excluded from decay); https://arxiv.org/html/2510.15262v1 (parameter classes);
https://arxiv.org/html/2411.18704v1 (multiple EMAs on one trajectory).
These are hypotheses for GPTrans, not established molecular-model improvements.

## Fixed recipe and two arms

Both retain GPTrans-T/2 degree-BondLocal+fresh-pair-update-norm, initialization43,
sampler/dropout42, physicalBS128,FP32/noTF32, LR0.001,4-pass warmup and
60-pass cosine to1e-6,500K train[0,500000),50K internal development[500000,550000),
best-development checkpoint selection,234360updates/29998080presentations.
No teacher, pretraining, geometry, altered membership or extra baseline arm.

A norm_no_decay: backbone and all other trainable parameters retain WD0.05;
only affine weight/bias owned by normalization modules have WD0. No blanket
exemption for other biases or embeddings. EMA0.999. This isolates decay on
normalization parameters; normalization type/position/forward remain fixed.

B multi_ema: all parameters retain WD0.05. One live model and one optimizer
trajectory maintain two independent EMAs updated at the same per-step cadence:
0.999 and0.9999. Primary predictor is pre-registeredEMA0.9999; EMA0.999 is the
within-trajectory mechanism comparator, not another trained arm. Each has an
independently selected best-development state with explicitly named metrics.
No averaging coefficients selected after the results and no extra training views.

## References and interpretation

Retained strict reference: ../pcqm_gptrans_pairnorm_init_noise_500k/runs/
pcqm-gptrans-pairnorm-init-noise-500k-s43/prospective/strict_qualification/
01674f002e108000c6cc/init_s43/reference_bundle.json.
Recipe: ../pcqm_gptrans_pairnorm_init_noise_500k/init_s43/recipe.json.
Initialization: D:/molgap-artifacts/initializations/gptrans-pairnorm-init-noise-500k/s43.pt.
Reference primary MAE is about0.1010866eV. Bind its exact reference identities
through existing prelaunch qualification before claiming a strict historical effect.

A primary effect is versus that retained reference; B primary effect is its
EMA0.9999 versus EMA0.999 along the same live trajectory. Historical comparisons
for B are supplementary. A versus B changes multiple mechanisms and must not be
called a single-variable causal contrast.
A candidate nomination requires primary delta<=-0.003eV and paired-row95%CI
upper<0 on50000 aligned finite predictions,2000resamples/seed42. Report each
candidate separately; do not require both directions to win. These are two
exploratory hypotheses; report both results, no winner-only presentation.
Intervals condition on rows and selected models, not training-seed variation.
If reference qualification or auxiliary EMA evidence is unavailable, preserve
pending/inconclusive status rather than fabricate a comparator or retrain it.

## Resource and replay contract

One Kaggle1 TeslaT4x2 session,one live training arm per device; expected about
10-12wall-hours, ceiling42900seconds/180seconds cleanup reserve and23.833334
allocatedT4-device-hours. Extra EMA evaluation overhead remains unmeasured.
No automatic retry or continuation. Official validation/test_dev/test_challenge
stay sealed. Existing epoch logger,source packaging,receipts,native qualification,
minimal retrieval,draft-closure,strict acceptance and terminal RML owners apply.

Durability must retain both EMA states and update counters in atomic resume;
resumed EMA histories and selected endpoints must match the declared semantics.
Primary selected models,aligned predictions,60-pass traces,optimizer/resume state,
observed roles and native costs are mandatory independently for each arm.
B also retains the named auxiliary EMA comparator predictions/selection history
required for its within-trajectory claim. Do not download all epoch checkpoints.
Two primary arm replay entries must be complete without exclusions before calling
the experiment replay-ready. No full-training release or automatic adoption.
