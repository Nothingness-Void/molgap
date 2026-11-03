# Norm decay and EMA100K: prospective discriminating screen

Authorized desktop Kaggle1 screen,2026-11-04. Base74fd7d61, new infrastructure reused.

## Incremental-information audit

The accepted BondLocal+PairNorm100K output has one EMA0.999, selected model,
last checkpoint at46860steps and native qualification probe checkpoints. The
last checkpoint metadata contains only one EMA tensor mapping, not a multi-EMA
bank. There is no complete per-step live-weight sequence in its retained manifest.
Another EMA recurrence cannot be exactly reconstructed from selected/final weights.
Changing normalization decay in the optimizer is a training intervention; a
post-hoc parameter edit cannot answer it. No model forward or data-role inference
was needed for this audit. Existing evidence rejects global WD increase, not
normalization-only exemption. Thus both proposed discriminators add information.

## Two live trajectories, four named predictors

Both arms use accepted GPTrans-T/2 degree-BondLocal+pair-update-norm,
initialization42, sampler/dropout42, physicalBS128/drop-last, FP32/noTF32,
LR1e-3,WD0.05,4-pass warmup/cosine60 to1e-6. Primary EMA0.999 in BOTH arms;
auxiliary EMA0.9999 in BOTH arms. The only between-arm training difference is
LayerNorm affine weight/bias zero-decay in norm_no_decay_multi_ema; all other
parameters, including ordinary linear biases and embeddings, retain WD0.05.
The normalization type, placement and forward are unchanged.

uniform_wd_multi_ema is the same-job reference AND acquires the previously absent
long-EMA predictor on its exact live trajectory. No third baseline-only training,
extra seeds, teacher or ensemble experts. The historical score0.139718318eV is
context; the same-job contrast is the primary causal normalization experiment.
The old500K draft was never submitted and is superseded for execution.

## Roles, exposure and gates

Frozen100K train source_idx[0,100000),50K selection-used development[100000,150000),
46860updates/5998080presentations/60passes per arm. Official validation,test_dev,
test_challenge remain sealed. Preserve exact existing fixed-cache identities.

Primary normalization delta is exempt-minus-uniform at separately selected
best-development EMA0.999 endpoints. Gate:<=-0.003eV and upper paired-row95%CI<0,
2000resamples,seed42. Nomination is one-seed100K only, not500K transfer.

Two predeclared secondary contrasts: auxiliary0.9999 minus primary0.999 on EACH
live trajectory, at independently selected endpoints. Report delta,conditional
paired-row interval and3meV gate for each; these are descriptive same-trajectory
EMA questions, not separate strict RML arms. Do not choose the best decay after
viewing results as the primary score. Report normalization at0.9999 and the
difference of the two EMA effects as exploratory interaction descriptions.
No unadjusted secondary interval establishes a new independent-test claim.

A numerical null still informs the decision: reject this normalization exemption
or EMA window as a material improvement under this contract. Opposite EMA effects
indicate interaction, not a universal better EMA. Missing auxiliary/row/state
identity is inconclusive; never manufacture a comparator or promote from loss alone.
No early-stop rule from partial curves; no automatic follow-up grid or scale-up.

## Trace, cost and acceptance

Reuse shared epoch logs,optimizer/EMA owners,atomic resume,independent auxiliary
selection/predictions and canonical curves. Do not reset or omit either EMA on
resume. Record passes10/20/30/40/50/60,selected/final gaps and train metric semantics;
online live normalized TRAIN is not fixed-weight EMA training MAE.
One T4x2 allocation,cap16200wall-seconds,180cleanup reserve,9T4-device-hour ceiling.
Expected3.5wall-hours is an estimate including extra EMA evaluation, not measured.
Luna owns status checks. On cap preserve actual partial state, no automatic resume.

Use submit_workflow,strict minimal retrieval,draft-closure,accept_run and native
qualification. Require two independent complete primary replay entries with no
exclusions,plus auxiliary state/selection/prediction/trace retention for EMA claims.
Download only manifest-required proof/model/prediction/trace/resume files to D:.
No new tests,manual acceptance schema or infrastructure edits are introduced.

Sources: ../pcqm_gptrans_bondlocal_pairnorm_100k/qualification.md;
../pcqm_gptrans_pairnorm_wd_500k_s43/analysis_20261103.md; companion s44 decision;
https://arxiv.org/html/2405.13698v3 and https://arxiv.org/html/2411.18704v1.
