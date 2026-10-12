# Triplet follow-up dual100K — 2026-10-13

Server Track C; MOLGAP-COMMON-V5-FINAL. Authority: the user requested one
dual-arm Kaggle2 task after the saved-trajectory/literature review. Exactly two
independent seed42 candidates, one T4 each. No baseline retraining, geometry,
teacher, extra seed,500K/full training, protected role or automatic successor.

## Evidence and question

The [curve-level review](../pcqm_v5_route_portfolio/curve_review_2026-10-13.md)
found continued training-error reduction after the selected aggregate endpoint
without corresponding EMA-development improvement. The accepted
[source-dependence diagnostic](../pcqm_gptrans_source_dependence/decision_terminal.md)
did not establish broad source collapse or gate starvation. Therefore neither
source dropout nor another amplitude cap is justified. The accepted triplet
attention did not improve on cheaper aggregation. Its uncertain later-cohort
gain is not proof of500K transfer.

[TGT](https://arxiv.org/html/2402.04538v2) motivates existing inward/outward
third-pair routing; this experiment retains it. Channelwise multiplication
tests a relation-composition hypothesis, not a reproduction of a published
score, a claim of3-WL expressivity, or geometry inferred from topology.

## One intervention per arm

- A, `degree_local_triplet_drop_ema999`: preserve all aggregate parameters and
  routing. During training only, independently drop the entire additional
  triplet return for each molecule with probability0.1, rescaling survivors
  by1/0.9. Parent local/GPA branches and their dropout RNG are untouched.
  Each insertion owns a layer-seeded private CPU generator with its state in
  a persistent uint8 buffer, preserved by native checkpoints. Evaluation is
  identical to unregularized aggregate inference. Parameters:5,880,961.
  Falsifier: selected-development gain and late-curve divergence do not improve;
  regularization may merely slow fitting or remove useful communication.
- B, `degree_local_triplet_product_ema999`: preserve scalar third-pair
  bias/gate, normalization, two8-channel heads and inward/outward directions.
  Replace scalar-weighted `V(value-pair)` by scalar-weighted
  `U(third-pair) * V(value-pair)` before summation. Only two32to16 projections
  per insertion are added; parameters:5,885,185. Dense contractions retain
  quadratic intermediate storage and cubic arithmetic. No cubic attention
  score tensor or geometric input. Falsifier: compositional capacity overfits
  or costs more without improving selected predictions.

Both retain blocks3/6/9/12, all valid/virtual pairs, the unchanged readout,
zero initial return and every accepted aggregate **scratch initialization**
tensor. No trained-weight initialization. All new projections are independently
layer-seeded; masks do not consume the parent's RNG stream.

## Frozen scientific contract and reference

Enroll the accepted aggregate terminal candidate as an additive reference view
using `enroll_terminal_candidate_reference`: exact checkpoint, predictions,
canonical trace, runtime, cost, roles, acceptance and source pointers must
verify before compute release. No scalar-only reference or new baseline run.
Compare each arm independently to that reference, not only to one another.

Use fixed cross-platform100K assets: train[0,100000), development[100000,150000),
OGB atom9/bond3/SPDcap20; direct normalized Gap-L1, fixed train-only transform;
FP32/noTF32, seed42, physicalBS128, accumulation1, drop-last32;
AdamW lr0.001/wd0.05/single-group/foreach=false/fused=false, clip1;
original dropout/drop-path0.1;60passes, warmup4, cosine to1e-6;
EMA0.999 each step, selected minimum EMA-development MAE. Each arm completes
46,860updates and5,998,080presentations. No recipe or data change to fit a new
runner. The native qualified family owner is reused because the newly integrated
shared kernel's foreach optimizer is not this frozen reference recipe.

Predeclared incremental nomination gate: gain>=0.001eV with a positive paired
row-bootstrap95% lower bound. This is a bounded single-seed screening decision,
not a training-variation confidence interval or full-scale promotion. Below
gate remains unpromoted; no post-hoc relaxation. Historical0.003gates remain
unchanged. Source/contract equality is checked independently of outcomes.

## Reuse, cost and terminal acceptance

Reuse `freeze_followup`, `prepare-release`/`check-release`, source/bootstrap,
T4-isolated workers, native optimizer preflight, training/checkpoint/chunk IO,
acceptance, per-arm RML transaction and existing A/B monitor. The only model
delta is `molgap.gptrans_triplet_followup`; adapters register its two modes.

Estimate5wall/10allocatedT4hours, ceiling8wall/16allocatedT4hours. Native
preflight must retain15% VRAM headroom and fit its5.5hour training estimate.
Resource/budget failure preserves artifacts, never changes BS/precision/recipe
or silently retries. Account quota is not API-certified.

Before formal training, preflight both arms including forward/backward,
optimizer/EMA, finite states, gradient connectivity, repeated determinism and
checkpoint binding. Save atomic last/best checkpoints each epoch and durable
chunks every10epochs. Retain native and canonical curves, live/EMA development,
per-insertion RMS/gradient/weight telemetry, selected predictions/targets/rows,
runtime/source/initial/transform hashes, observed role events and actual cost.
Native independent saved-output acceptance then `close_author_outputs` must
produce two complete candidate/reference Replay admissions before any
Replay-ready claim. Do not fabricate missing artifacts or reclassify errors.

This task uses only its frozen train/development roles. A later500K-cohort
inference audit needs a separate explicit NO_TRAIN plan with accepted-parent
hashes; its absence does not establish transfer. No official validation,
test-dev or test-challenge access. Existing Luna B monitors only the returned
ID/version every30minutes: healthy silent, one idempotent event to existing A
for completion/actionable fault, UNKNOWN for API uncertainty. No new chat,
new cron, A-model override, desktop custody or automatic successor.
