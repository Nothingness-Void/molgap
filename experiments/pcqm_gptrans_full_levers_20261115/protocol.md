# Full-scale GPT-T lever experiments (frozen 2026-11-15, before any result of these runs)

Authority: user, 2026-11-15 (all changes authorised; IMS submission authorised). Method: planner decision 2026-11-15 (Opus). Branch `codex/exp/gptrans-full-levers-20261115`, based on `codex/exp/gptrans-full-minus-tail-20261110` (e32009ae). Base recipe: [recipe_frozen_gptrans.md](../pcqm_full_minus_tail_fusion_20261113/recipe_frozen_gptrans.md), unchanged except for the one lever each arm names. Prior question: [decision_full_minus_tail.md](../pcqm_full_minus_tail_fusion_20261113/decision_full_minus_tail.md).

This file authorises no protected-role access. Official valid/test and sealed labels are not read. `tail_holdout_v1` rows (3,250,000..3,378,605) never enter training; every endpoint is scored on tail exactly once. The production registry is not touched. No IMS smoke or preflight job is ever submitted (queue rule); checks run locally and as early exits inside the formal job.

## Question
At full scale (rows [0, 3,250,000), 60 passes) GPT-T B1024 reaches 87.941 / 87.846 meV tail MAE (s42 / s43; two-init equal-weight ensemble 85.349 meV). Which cheap lever lowers tail MAE beyond init noise: more inits in the ensemble (A), smaller batch (B), more depth (C), longer training (D), or a cross-recipe ensemble (E)?

## Arms

Common to every arm unless the row says otherwise: GPT-T BondLocal + PairNorm, AdamW wd 0.05, clip 1.0, 60 passes (warmup 4, cosine to 1e-6), TF32, no compile, rows [0, 3,250,000) (`--max-shards 65`), endpoint = last-pass EMA, runner `platforms/colab/pcqm_k1_b1024_production/train_full_b1024.py` sha256 `120d5571593c56647afe8f7c6f6060afeb7781ff8aea101fb9454fdac0ee5b37`, IMS A100, jsub -q H (12 h walltime, resume from `last_checkpoint.pt` by resubmitting the same command). As in the s43 precedent, only the init file changes between inits; sampler, dropout and global RNG stay seed 42.

| Arm | Change | Runs | Trajectory ids (`TB-pcqm-gptrans-full-levers-20261115-<arm>`) | Status |
|---|---|---|---|---|
| A | B1024, EMA 0.992028 (= 0.999^8), lr-mult 2.82843 (sqrt 8), new inits only | s44, s45 | `a_b1024_s44`, `a_b1024_s45` | submit now |
| B256 | `--gptrans-batch-size 256`, `--gptrans-lr-mult 1.41421` (sqrt 2; runner lr = 1e-3 x mult, so 1.41421e-3), EMA 0.998001 (= 0.999^2), warmup 4 passes, 60 passes, 12,695 steps/pass, 761,700 steps | s42, s43 | `b256_s42`, `b256_s43` | other agent, after local throughput |
| B512 | as B256 with batch 512, lr-mult 2.0 (2.0e-3), EMA 0.996006 (= 0.999^4), 6,347 steps/pass, 380,820 steps | s42, s43 | `b512_s42`, `b512_s43` | only if B256 is replaced (rule below) |
| C | B1024, depth 12 -> 20 layers, all else as A; s42 init generated locally for 20 layers | s42 | `c_l20_s42` | other agent |
| D | B1024, 90 passes (TOTAL_EPOCHS constant 90; warmup 4, cosine over 90), s42 init (same file as the s42 member) | s42 | `d_p90_s42` | conditional (rule below) |
| E | cross-recipe equal-weight ensemble, scoring only | none | `e_cross_recipe` | no new training |

Seeds/init files: s42 `gptrans_init.pt` (file sha `27590c1a...c3320`, tensor-state sha `a65034da...d1b`); s43 `gptrans_init_s43.pt` (file `2ccd1eb79de4b82fffb58522b9e2325026d36f04f398be0d49d3eea55ee05f3c`, state `c95da412764ea1b9a701ae9421c167e32a18f9dc423d364aa3d4fe8c0a8d5f5b`); s44 file `2415cdbfea332db158086b2ee457ce95bcb90a2e3f87c04564146ba12a2c3573`, state `ff9496504ff111e0fb94b23a2d62930ebb1d0c027cdd06a0df9697b760d1728b`; s45 file `f653837b8f8e7e64531beaa103dd161d5feafb169e9c954bbe788c9c7e412486`, state `58c48765ec5892897b572506ef6da13503ebf51b7a33a306690ed6470ce26b7b` (all under `D:/molgap-artifacts/initializations/gptrans-pairnorm-init-noise-500k/`; sha re-checked locally 2026-11-15 and again on the remote after upload).

### Conditional rules, fixed now
- **B256 vs B512.** Measure local training throughput for the B256 recipe, extrapolate to IMS A100 wall time using the measured B1024 IMS rate (6.3 step/s at B1024, 8.4-8.6 h per 60 passes). If the extrapolated total wall time exceeds 20 h, use B512 (not B256) for both inits; otherwise B256. The choice is recorded before submission and uses no tail number. The unchosen arm ends `NO_TRAIN`. Estimated native wall: B1024 about 8.5 h (measured); B256 and B512 not yet measured (the extrapolation above decides); a run over 12 h resumes in further 12 h jobs.
- **C (depth 20).** Layer count 12 -> 20 in the init builders (`src/molgap/family_model_ops.py`, `src/molgap/gptrans_bond_local.py`). Architecture is not part of the config hash, so the protection is the init sha: the 20-layer seed-42 init is generated locally with the same ogb/torch builder, its file sha and tensor-state sha are recorded in the submission record, and the run record states `layers=20`. Parameter count about 1.5-2x. Estimated wall about 13-17 h (not measured).
- **D (90 passes).** Submit only if the offline curves in `D:/molgap-artifacts/offline_curves_20261115/summary.md` show the pass 50-60 probe falling by at least 0.5 meV per 10 passes. If that file does not exist or the drop is smaller, D ends `NO_TRAIN`. Estimated wall about 12.8 h (90/60 x 8.5 h, not measured).

Expected IMS remote roots (`/lustre/home/users/sm2/chou/`), run directory under each root `runs/<name>/`:

| Arm | Remote root | Run dir name |
|---|---|---|
| A s44 | `molgap-gptrans-minus-tail-b1024-s44-20261115` | `gptrans-b1024-minus-tail-s44-20261115` |
| A s45 | `molgap-gptrans-minus-tail-b1024-s45-20261115` | `gptrans-b1024-minus-tail-s45-20261115` |
| B256 s42/s43 | `molgap-gptrans-minus-tail-b256-s42-20261115` / `-s43-` | `gptrans-b256-minus-tail-s4x-20261115` |
| B512 s42/s43 | `molgap-gptrans-minus-tail-b512-s42-20261115` / `-s43-` | `gptrans-b512-minus-tail-s4x-20261115` |
| C | `molgap-gptrans-minus-tail-b1024-l20-s42-20261115` | `gptrans-b1024-l20-minus-tail-s42-20261115` |
| D | `molgap-gptrans-minus-tail-b1024-p90-s42-20261115` | `gptrans-b1024-p90-minus-tail-s42-20261115` |

Run ids in RML: `pcqm-gptrans-full-levers-20261115:<arm>`. Data for every root is a read-only symlink to the s42 root's `data/ogb-train-full` (realpath inside the chou tree).

## Noise floor and sanity band (A members)
Noise floor = standard deviation (ddof 1) and range of the four B1024 inits' tail MAE (s42, s43 known; s44, s45 from A). N = max(1.08 meV, range). 1.08 meV is the 500K six-init range.
Each B1024 single-model tail MAE must fall inside 87.89 +/- 1.5 meV (87.89 = mean of s42 87.941 and s43 87.846). A member outside the band is recorded INCONCLUSIVE; infrastructure is checked first; it is not rerun and not removed from the ensemble. The band applies to same-recipe inits only; B, C and D change the recipe, so only their own criteria below apply.

## Pre-registered criteria
Tail MAE in meV, equal weights, nothing fitted. Paired row bootstrap on tail rows: 2000 resamples, numpy `default_rng(42)`, the same index sets for every comparison. Five comparisons (A, B, C, D, E), Bonferroni: a single one-sided 99% upper bound (99th percentile of the bootstrap difference) per comparison. sigma = sample standard deviation of the four B1024 tail MAEs; M = their mean.

- **A (same-recipe all-init ensemble).** Members: every completed B1024 init (s42, s43, s44, s45), equal weights, none removed, no reweighting, decided before s44/s45 are scored. PASS iff (4-member ensemble - 85.349) <= -1.0 meV and the 99% upper bound < 0. 85.349 is the s42 + s43 equal-weight tail MAE, a number chosen post hoc after seeing tail (one of four pairs). It is a known baseline, not an unbiased reference. PASS makes "all B1024 inits, equal weights" a candidate.
- **B (batch size).** PASS iff mean(B-arm tail s42, s43) <= M - max(1.5, 3 sigma) AND, for each init (B s42 vs B1024 s42; B s43 vs B1024 s43), the paired difference has 99% upper bound < 0.
- **C (depth 20).** PASS iff tail(C) <= M - max(1.5, 3 sigma) and the paired difference against B1024 s42 has 99% upper bound < 0. Descriptive only: whether the gap between the in-train probe (20,480 rows) and tail widens versus B1024 s42.
- **D (90 passes).** Same criterion as C (against B1024 s42).
- **E (cross-recipe ensemble).** Members: every completed run of every arm and the B1024 inits, equal weights per run, none removed. Compared with the best same-recipe ensemble (the lowest tail MAE among: all B1024 inits; all B-arm inits). PASS iff (E - best same-recipe ensemble) <= -1.0 meV and the 99% upper bound < 0. Otherwise E is not adopted. Costs nothing extra.

Outcome vocabulary: PASS / FAIL, INCONCLUSIVE (member outside the band, missing or incompatible endpoint), NO_TRAIN (arm not submitted by rule). No snapshot is selected inside a job; no rerun; no change of weights, members or thresholds after any tail number is seen. A tail score is computed once per endpoint, with `platforms/local/pcqm_tail_holdout_eval.py --run-config` (as for the s42/s43 members). The verdict code is committed before any new tail number is computed.

## Stated limits
- 85.349 is a post-hoc 1-of-4 number (see A).
- Deciding between levers by tail PASS/FAIL follows the 2026-11-15 precedent. The more candidates are compared on tail, the more optimistic the final candidate's tail number is. The final candidate's tail MAE is not an unbiased estimate.
- Official valid has not been read. Any official-valid report on a final candidate needs a separate explicit statement from the user.
- Init noise: single runs per lever except A and B; C and D are single-init and compared against a four-init mean, so the sigma-based bar carries the noise.
- Platform: all arms run on IMS A100, torch 2.7.1 (same as the s42/s43 members), so hardware is not a confound within this question.

## Cost
Native units kept separate. A100 hours (estimated, not measured): A about 17 (2 x 8.5), B unmeasured, C about 13-17, D about 12.8 if run. Credits: `showlim -c` group ew6 remaining about 41,500 points at the start; each B1024 job about 100 points (estimate: 8 cores x 12 h bound). Measured values are filled after each run from `sessions.jsonl`; unmeasured stays missing.

## Records
Prospective trajectories, one per arm, are published under `prospective/` in this directory (policy `pcqm-gptrans-full-levers-20261115`, role plan [role_plan.json](role_plan.json), decision stub [decision.md](decision.md)). A trajectory authorises no training. Submission records: [ims_submission_levers.md](ims_submission_levers.md).
