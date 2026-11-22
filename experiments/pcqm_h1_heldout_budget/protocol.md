# H1 fixed-budget pilot and N4 replication

Track B, local only, 2026-11-22. The user approved follow-up research after
[P0](../pcqm_pairnorm_residual_2d_p0/decision.md). P0 found chemistry signal
on checkpoint-selection rows, not a gain after paying for calibration labels.
This study tests that missing comparison at **100K**, not 3.25M. A 6% leave-out
approximates the proposed 200K/3.25M fraction. No remote job, tail, official
valid/test, 3D input, model promotion, second seed or larger run is released.

## Models and roles

- Anchor: RTX 5060 `local-pcqm-gptrans-100k-20261119-s42/v1/output-native`,
  `pair_update_norm`, selected EMA .999. Immutable output manifest
  `bd0ebc8d6c15639abf7627b94dea89709c4c12a58ff206a51277b7d27b3ca007`.
  Its original run took 75.71 min. Reinspect the existing package/receipt, do
  not retrain the anchor or promote its mechanical inspection to strict replay.
- Candidate: use the registered local family trainer. Change the backbone
  train prefix from `[0,100000)` to `[0,94000)`; calculate normalization on
  those 94K only. Same init `27590c1a928cafa8b633998e2291916b8f171d345ed5cb3b0234f0e9a37c3320`,
  seed 42, batch 128, FP32 without TF32, AdamW, warmup 4 / cosine 60 passes,
  best-development EMA .999. Same checkpoint-selection rows `[100000,150000)`.
  The candidate gets 44,040 rather than 46,860 updates by design. The data cost
  includes the shorter exposure; this is not a causal estimate per lost row.
- The `[94000,100000)` labels may fit the corrector, not select, normalize or
  train the candidate backbone. Make a topology-only prefix of the original
  accepted 100K packed data. No protected row is needed. Use the original P0
  descriptors, counts, Morgan features, missing-value policy, `make_gbm`,
  scalar candidates, inner scaffold split and tie order unchanged. Refit the
  selected scalar control and the chemistry corrector on all 6,000 fit rows.
- One fixed external-to-these-models block: **all `[550000,600000)`**. It is
  part of OGB's training split, not a globally blind test. None of the four
  checkpoints used it for training or checkpoint selection. It lies beyond
  P0's discovery cohort. The primary estimand includes scaffold recurrence;
  report scores also on scaffolds absent from the 6K and count canonical
  strings shared with the 6K. Keep invalid rows in the primary comparison.
- N4: freeze the same s43/s44 500K selected checkpoints as P0, not their best
  member. Compare neither scale's absolute MAE with the historical tail score.
  Check each of the four checkpoints against the first 128 of its own saved
  development predictions (max absolute error <= 0.0001 eV). A model, ID,
  target, hash or role mismatch blocks interpretation; never substitute an arm.

## Endpoints fixed before reading the new block

Score `[550000,600000)` once, after sealing all four backbone predictions and
both correction outputs with row IDs. Score labels may be decoded incidentally
with official-train graphs; do not feed them to predictors or inspect their
values/metrics until prediction hashes are sealed. Independently align the
archived CSV labels with the packed graph labels before computing metrics.

Report raw 100K, raw 94K, 94K + scalar, 94K + chemistry MAE (meV), comparisons
on identical rows. Let N = chemistry minus raw 100K (net), R = chemistry
minus raw 94K (rescue), F = chemistry minus scalar, C = raw 94K minus raw
100K (cost). Require N = C + R. Each interval is the repository's paired row
bootstrap, 2,000 draws, seed 20261122, fixed models. PASS only if N and R are
both <= -1.5 meV with upper 95% bounds < 0 and F has upper bound < 0. FAIL
if any lower bound for N or R is > -1.5, or F's lower bound is >= 0; otherwise
INCONCLUSIVE. No arm selection on the score block. One init, one contiguous
calibration block; this is not a random sampling of training regimes.

N4 replication uses only the raw 500K pair. Freeze the one group found by P0:
RDKit parses and every bond is single. Require each init's group/global MAE
ratio >= 1.5 and fraction of total absolute error >= 0.10, and same-sign
co-miss (both errors >= 150 meV in magnitude) prevalence >= 1.5x global.
Report all three even when a gate fails. As a descriptive check, compare
single-bond vs other valid molecules after matching the distribution of
heavy atom bins (<=10, 11-15, >=16), ring bins (0, 1, 2, >=3) and true Gap
bins (<4, 4-6, 6-8, >=8 eV). Use only strata with >=20 rows per group,
weight by the single-bond counts there, and report coverage. Not a causal
claim or subgroup-training release.

## Custody and cost

The study trajectory precedes any data preparation. The derived worker recipe
pins actual train statistics and shard bytes; the local preparation owner
publishes its executable trajectory before training. The study ledger records
analysis/inference only; the worker owns training cost. Budget: one local
60-pass training, expected 1-2 h, plus at most one CPU fit and four frozen
endpoints, expected <30 min analysis. No grid or performance-driven retry.
Historical 100K and 500K artifacts are sunk cost. No papers are needed to
decide this predeclared comparison.
