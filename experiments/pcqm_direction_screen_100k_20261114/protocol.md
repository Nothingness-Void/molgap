# Direction Screen, 100K, One 12-Hour IMS Round (Prospective Protocol)

Prospective record, 2026-11-14. Owner: server (IMS A100). Source design: [DESIGN_opus.md](DESIGN_opus.md)
(Opus, Chinese), superseded where this file differs by the Opus final revision relayed by the coordinator
(scope: 7 families, one trajectory per arm; the revision text itself is not in the repository, so this file is the
retained statement of it). This record authorizes no submission and no training. It is published before any
diagnostic or training, as AGENTS.md requires.

## Question

At the K1 100K scale (`pcqm_fixed_100k_v1`, pure 2D), which paradigm-level levers move 50K development gap MAE by more
than init noise, and is the problem data-limited or capacity-limited? Observed deficiency: K1 module-level 100K tests were
repeatedly NULL, the only positive (Muon) faded at 500K, and no same-frame measurement of data-size slope, capacity proxy,
budget, objective/sampling, cheap baselines or residual structure exists. Decision changed: choose between a data/ensemble/scale
path and one paradigm lever worth a second seed plus an attenuation check, or conclude that none is worth further 100K module search.

## RML evidence reviewed (queries run 2026-11-14 with `--repo-root` pinned to this worktree)

Queries: `lookup --mechanism` for scale, width, depth, capacity, epoch, exposure, lr, wd, decay, loss, huber, sampling, oversample,
weight, gin, fingerprint, baseline, aux, auxiliary, descriptor, muon, ensemble, tail, data-size, scaling; `lookup --family`
scale-fit, init-noise, wd5e2; `cards --term gine`; `gaps`. No card exists for: depth, capacity, huber, sampling, oversample, data-size, scaling.

| Evidence (trajectory/family) | Outcome and fact | Why it does not cover this screen |
|---|---|---|
| `pcqm-k1-init-noise-100k` (s43/s44) | D = 1.631 meV init spread; 3-init ensemble gain 7.683 meV; residual corr 0.848; same-seed bit-identical on T4 | Kaggle T4 only; the noise floor must be re-measured on A100 (D7) |
| `pcqm-k1-lr2e4-100k`, `-lr8e4-warmup-100k`, `-wd5e2-100k` | NEGATIVE (m = +1.30, +0.39, +0.62 meV) | Only lr 2e-4, 8e-4, wd 5e-2; here lr 1e-4, 1.6e-3, wd 0.1 probe the basin width |
| `k1-slot-width96` (NEGATIVE, below_gate); historical node-width256 | slot/node width only | K1 trunk width/depth is deferred; width and depth are probed on GINE (D2) as a proxy, not as K1 evidence |
| `pcqm-k1-muon-100k-20261106` | POSITIVE -8.77 meV, `early_gain_fades` | Optimizer paradigm; its fade means any positive here needs an attenuation check. SOAP (D3) is a different optimizer, untested |
| `pcqm-k1-short-ensemble-100k` | NEGATIVE (equal-compute 2-member) | Compute split, not epoch budget (20 vs 80) |
| `pcqm-scale-fit-retained`, `pcqm-scale-transfer-*` | NO_TRAIN / cross-scale diagnostics | No same-contract N-curve at 12.5K-100K; no train-vs-dev gap per N |
| `gptrans-clean-chemical-descriptor-aux` / `-fingerprint-aux` (v3, v4) | INCONCLUSIVE | GPT-T aux; K1-aux is deferred |
| `gptrans-t` noisy-nodes (100K POSITIVE, 500K NEGATIVE) | lost_at_scale | Precedent for the attenuation requirement |
| `pcqm-k1-clean-second-100k`, `pcqm-k1-sam-100k` (+2.83), tail-regularization 500K | consistency/regularizer negatives | No point-loss (MSE, Huber, binned, Laplace) or density/subgroup sampling test exists |
| `gine-gap-specialist` (GINE 1M) | POSITIVE specialist, no paired effect | Different data (1M, scaffold dev) and role; no same-frame GINE-size/objective reference. FP-MLP and SMILES-T: no record |

`gaps`: `gine-gap-specialist` is sparse (n=1, no effect); no contradiction entry concerns these mechanisms. Uncarded history
matched node-width256 and clean-fingerprint-qualification only. None of the questions is answered by existing evidence, so
DUPLICATE_EVIDENCE does not apply. Noise-only and descriptive outcomes remain valid terminals.

## Families and arms (one prospective trajectory per arm, 37 trajectories)

Trajectory ids are `T-pcqm-ds-<family code>-<arm>`; family ids are `pcqm-direction-<code>-20261114`.

| Family | Arms | Falsifier (cheapest decision-relevant) |
|---|---|---|
| D1 data curve | `k1_n12500`, `k1_n25000`, `k1_n50000` (K1 thin wrapper, nested first-N rows, 40 epochs); `gine_n{12500,25000,50000,100000}`; `fpmlp_n{12500,25000,50000,100000}` | Flat K1 slope or small train-dev gap refutes data limit; cheap-model slopes give shape reference |
| D2 capacity proxy (GINE) | `gine_h96_l6`, `gine_h384_l6`, `gine_h192_l12` (h192/6 layers = D1 `gine_n100000`) | GINE gains inside noise at 2x width/depth refute capacity limit as a proxy |
| D3 training budget (family worker, K1) | `k1_ep20`, `k1_ep80`, `k1_lr1e4`, `k1_lr1p6e3`, `k1_wd0p1`, `k1_soap` | All inside the noise floor refutes optimization limit |
| D4 objective/sampling (K1 and GINE) | `{k1,gine}_{mse,huber,binhyb,laplace,lds,oversample}` | No overall signal and no tail-bin bias change refutes loss/sampling-induced regression to the mean |
| D5 reference | `smilest_n100000` (FP-MLP and GINE references are D1 `fpmlp_n100000`, `gine_n100000`, not duplicated) | Position only, descriptive |
| D7 ensemble noise (K1) | `k1_s45`, `k1_s46`, `k1_s42_a100` | Sets the in-frame init range; no gate |
| D8 analysis only | `d8_residual_lgbm`: descriptor LightGBM residual model on K1 train-row predictions | CPU only; residual model with no held-out gain refutes descriptor-recoverable residual |

Deferred (listed, no trajectory, not authorized): K1 width/depth, K1-aux, readout/descriptor readout, SSL pretraining,
attention micro-tuning (QK-Norm, gating), 3D distillation, Muon, GCN-VN, any C-batch arm (second seeds, 160 epochs).
Conditional: an N200K point (rows 150,000..249,999 of the 500K export) is run only if that path exists on IMS; it gets its own
trajectory at that time and, being outside rows 0..99,999, needs an explicit role note.

Assumptions of this file: GINE is the 6-layer h192 virtual-node GINE unless the arm changes h/layers; GINE and FP-MLP N arms use
nested first-N rows; D2 and D5 reuse D1 100K arms instead of duplicate runs; D8 needs K1 predictions on all train rows, which
the DESIGN train probe (5,000 rows) does not give. If those are not retained, D8 closes INCONCLUSIVE or NO_TRAIN with missing_discriminator.

## Unified protocol

B128, drop_last, strict FP32 (TF32 off), AdamW (lr 4e-4, wd 1e-5 unless the arm changes it), per-epoch cosine to 1e-6, grad clip 1.0,
L1 on the standardized target unless the arm changes it, target mean/std from the arm's training rows only, 40 epochs unless the arm
changes it. Dev evaluation every 5 epochs and at the end on the 50K dev rows is monitoring only. Primary metric: final-live dev MAE;
no snapshot is selected on dev. Per-arm outputs follow DESIGN section C (`config.json`, `env.json`, `curve.json`, `dev_pred_final.npz`,
`train_probe_final.npz`, `timing.json`, `done.json` with sha256; atomic writes; resumable). Code goes under
`experiments/pcqm_direction_screen_100k_20261114/` and `platforms/ims/direction_screen/`; `src/molgap` is read-only. Shards are pure 2D:
`pos`, `edge_distance`, wedge fields, `geometry_valid`, `mmff_converged` are dropped and asserted absent; no 3D is constructed.

## Roles

| Role | Rows | Use |
|---|---|---|
| train | 0..99,999 (`pcqm_fixed_100k_v1`; N arms use nested prefixes) | training, target standardization, train probe |
| development | `fixed100k-next50k`, rows 100,000..149,999 | labels read for the metric; monitoring, no selection |
| `tail_holdout_v1` (rows 3,250,000..3,378,605) | not read | forbidden |
| official valid, `test_dev`, `test_challenge` | not read | forbidden |

Row identity (`source_idx` and target bytes sha256) is checked against the local manifest before any arm trains; a mismatch or any
protected row aborts the round. Heterogeneous-ensemble weights are fit on dev and therefore labelled `selection_used` and non-promotional.

## Hard constraints (AGENTS.md)

B3LYP gap in eV; pure 2D must not build 3D; tail holdout and official valid/test unread; IMS commands obey `platforms/REMOTE_HANDOFF.md`
and stay under `/lustre/home/users/sm2/chou/`; no `src/molgap` edit, no new tests; A100, CPU, wall and queue units separate; unknown is not
zero, estimated is not measured; no baseline retraining to fill a missing reference.

## Decision criteria (pre-fixed)

Baseline MAE = mean of the in-frame A100 K1 runs (`k1_s42_a100`, `k1_s45`, `k1_s46`); paired reference = `k1_s42_a100` (row bootstrap, 2,000
resamples, seed 42; row noise only). Noise floor `N = max(range of the six K1 inits, |in-frame s42 - s43|)`; the six inits are the retained
T4 s42/s43/s44 (D = 1.631 meV) plus the three A100 runs, and s43 is the retained T4 run since no s43 arm is scheduled.

- Overall signal: |delta| >= 5% of baseline, and >= 3 x N, and the paired 95% CI excludes 0.
- Subgroup signal (DESIGN C): delta >= 10%, subgroup n >= 2,000, CI excludes 0, overall not worse by more than 2%. Buckets: true-gap bins,
  radical, ring count, aromatic ring count (0 = none), heavy atoms, heteroatoms, charged.
- Tail signal: |bias| in the [2,4) eV or >= 8 eV gap bin falls by >= 30%.
- Data-limited: MAE(100K)/MAE(50K) <= 0.93 and train-dev gap > 30% of dev MAE.
- Capacity/optimization-limited: GINE h384 or l12, or K1 ep80, is >= 5% better than its baseline (GINE proxy is not K1 evidence).
- Anything else is descriptive. A positive is not a "direction" until a separately authorized 25K/50K/second-seed attenuation check passes.
- Scaling: per-doubling ratios and a descriptive a*N^-b fit; extrapolation to 500K or 3.38M is reference only.

## Output contract

`analyze_local.py` (local, after retrieval) writes per-arm/per-bucket MAE, signed bias, delta with paired bootstrap CI, scaling table, ensemble
curves and flags to `summary.json` and `SUMMARY.md`; contracts and hashes are checked before any number is trusted. Scheduler completion is not
scientific acceptance; outcomes are set only after this analysis.

## Cost events (planned, estimated, not measured)

U = one K1 100K 40-epoch run alone on one A100 (about 1.0 A100-hour; IMS TF32 run measured 3,553 s). A100 device-hours are DESIGN estimates for K1
arms only (N arms scale with N; ep20 0.5, ep80 2.0, others 1.0). GINE, FP-MLP, SMILES-T and SOAP arm costs are missing, not zero. Packing
4 arms per card with an unmeasured slowdown s = 1.6-2.5 means wall and queue hours are missing. CPU hours (local RDKit featurization, D8 LightGBM)
are a separate unit, also missing. Smoke measures s; measured costs are recorded at closure.

## Failure and stop rules

- Row identity mismatch or any protected row touched: stop everything, no training.
- Import/environment failure: one fix and one resubmission of smoke; a second failure stops.
- A model class fails smoke: drop that class, run the rest.
- Walltime truncation: resubmit once and resume. Single-arm NaN or OOM: record failed, no retry (an lr-extreme NaN is a result).
- Queue wait over 2 hours: add no jobs; shrink in order lr/wd extremes, cheap-model N points, then D7 seeds.
- If measured slowdown exceeds budget, cut arms; never change B128, precision or optimizer semantics for utilization.
- Valid terminal outcomes: NEGATIVE_UNDER_CONTRACT, INCONCLUSIVE, STOP_FOR_COST, NO_TRAIN. No 500K, full-scale or second-seed action follows
  without a new authorization.

## Limits of a single seed

Each non-baseline arm has one seed; init noise of about 1.6 meV (about 1.2% of 140 meV) limits detection to roughly >= 5%. Row bootstrap does not
measure training stochasticity. Subgroup findings are correlational. N arms change N and step count together. The A100 baseline may differ from the T4
history (141.3/139.7/140.6 meV); arms are compared only to the in-frame baseline.

## Trajectory structure

Per the revised scope, one prospective trajectory per arm (37), generated by a single `plan_many` batch against one frozen snapshot under
`prospective/<arm>/`, all citing one registered candidate policy `pcqm-direction-screen-100k-20261114` (RUN_PAIRED_100K or NO_TRAIN; not approved).
