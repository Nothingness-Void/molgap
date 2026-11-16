# GPT-T 500K capacity question: width and FFN (frozen 2026-11-17, before any result of these runs)

Authority: user, 2026-11-17, for preparing the IMS submission. A later agent reviews and submits. Branch `codex/exp/gptrans-width-500k-20261117`, from `molgap-desktop` 0e18f5c2. This file releases no protected role: official valid, test and the sealed 20K stay unread. `tail_holdout_v1` (rows 3,250,000..3,378,605) never enters training and each endpoint is scored there once. The production registry is not touched. No IMS smoke or preflight job is submitted (queue rule); checks ran locally and run again as early exits inside each task.

## Question
At 500K rows, does a wider GPT-T (width 256 -> 320) or a wider FFN (ratio 1 -> 3) lower tail MAE by at least 1.5 meV against the same-seed original (R)? A 20-layer arm calibrates what 500K can see.

## Evidence review
- Full scale (rows [0, 3,250,000), 60 passes): depth 20 gained 0.89 meV and 90 passes 0.81 meV, both below the 1.5 meV bar; batch 256 was worse; the cross-recipe ensemble failed ([decision_CD](../pcqm_gptrans_full_levers_20261115/decision_CD.md), [decision_BE](../pcqm_gptrans_full_levers_20261115/decision_BE.md) on branch `codex/exp/gptrans-full-levers-20261115`; summary in CURRENT_STATE). Capacity along depth is small at full scale. Width and FFN were not tried there.
- 500K, matched budget: 12x384 was not better than 12x256 and 16x256 was 0.71 meV better than 12x384, below the 3 meV gate ([terminal decision](../pcqm_single_model_total_budget/terminal_decision.md); development-row scale, one seed, noncausal; the same record puts 12x384 0.735 meV above the 12x256 reference).
- Gap in the evidence: no 500K tail comparison of width or FFN against a same-seed 12x256 exists, and no check that 500K can show a capacity gain at all. L20 is that check: its full-scale gain (0.89 meV) is known to be small, so a 500K gain near 0.9 meV or more means 500K carries the capacity signal.
- Cheapest falsifier: W320 and F3 each need a mean tail gain of at least 1.5 meV over R with both seeds negative and a one-sided 99% paired bound below 0. Missing any one of these ends the nomination.

## Arms
Every arm trains rows [0, 500,000) (`--max-shards 10`), GPT-T BondLocal + PairNorm, batch 1024, lr 2.828e-3 (multiplier 2.8284271, i.e. sqrt 8 of 1e-3), AdamW wd 0.05, clip 1.0, EMA 0.992028 (0.999^8), dropout 0.1, drop-path 0.1, 60 passes (488 steps per pass, 29,280 steps), warmup 4, cosine to 1e-6, endpoint final-ema, TF32 as in the runner, no compile, IMS runtime (torch 2.7.1 on A100). Learning rate, EMA, dropout and weight decay are the same in every arm: one factor changes per arm. Target statistics: the runner uses `GPTRANS_500K_STATS` = (5.413434769952119, 1.176728647311213) when the store holds exactly 500,000 rows; the launcher stops the run (exit 3) if the recipe holds other values, and `run_config.json` records them.

| Arm | Change vs R | Layers / width / heads / FFN ratio | Parameters |
|---|---|---|---|
| R | none (regression base) | 12 / 256 / 8 / 1 | 5,871,201 |
| L20 | depth 12 -> 20 (calibration) | 20 / 256 / 8 / 1 | 9,529,761 |
| W320 | width 256 -> 320, heads 8 -> 10 (head size stays 32) | 12 / 320 / 10 / 1 | 8,827,321 |
| F3 | FFN hidden 256 -> 768 | 12 / 256 / 8 / 3 | 9,023,073 |

Each arm runs with two init seeds, 42 and 43. As in the levers precedent, seed means only the init file: sampler order and dropout stay seed 42 (runner sha below has no `--seed`). The gap between R s42 and R s43 therefore understates run-to-run noise. Context: 500K B128 init scatter was about 0.47 meV SD on development rows ([init noise record](../pcqm_gptrans_pairnorm_init_noise_500k/decision.md)); the R seed difference is reported next to every verdict.

Inits ([init_record.json](init_record.json); built by [make_inits.py](make_inits.py) with the repo builder; the three existing files were checked, not rebuilt):

| Run | File | File sha256 | Tensor-state sha256 |
|---|---|---|---|
| R s42 | `gptrans_init.pt` | `27590c1a928cafa8b633998e2291916b8f171d345ed5cb3b0234f0e9a37c3320` | `a65034dae2d01d82eb0074eba8f2ee697d93ca259188ae45a4213e9041d1ed1b` |
| R s43 | `gptrans_init_s43.pt` | `2ccd1eb79de4b82fffb58522b9e2325026d36f04f398be0d49d3eea55ee05f3c` | `c95da412764ea1b9a701ae9421c167e32a18f9dc423d364aa3d4fe8c0a8d5f5b` |
| L20 s42 | `gptrans_init_l20_s42.pt` | `8ee2d6aa69168b451285aed872217b6bd7149ed81f7c6ed7c98beca035611b5e` | `15aa22a9a36606ef17ef74fb35d88347ed6bf3822fae297b0e260745bca085e9` |
| L20 s43 | `gptrans_init_l20_s43.pt` | `bb375fee040e239eaf37f116143834a23ebc1dab36f04d7d7081c86062ed3fe8` | `21b7de3f1fdaa13331a5f05318c3bde888d0ccea5f81aaccb658479358886340` |
| W320 s42 | `gptrans_init_w320_s42.pt` | `a395cd836a9bc3671cbe09a440242e6fe603e98d4642a21a51f306c6f4fbcce2` | `fda15782ca558f8d05f5ff3d9fe369238183ffa7eaf610826b70031586b56ef8` |
| W320 s43 | `gptrans_init_w320_s43.pt` | `a119dcc6b8ce7df43c26161e753c1cb5c9a787a716b0e2851d80f4315d4d601d` | `5a383a0315f9d93c7467fb0212a4ad487c3131ebc8c9a126825dc6b7ccac6d68` |
| F3 s42 | `gptrans_init_f3_s42.pt` | `f65a446da51a2a98f8bbe2c223690b263c358f284973adcdeae1fa1b30b3ff38` | `0f11b9aedbf7eab488bf2163c7838e177840335fa452361ac67b642b212f0cc6` |
| F3 s43 | `gptrans_init_f3_s43.pt` | `3d9756a38d4bb0c5ae0d205fad18c728b9b8e5badc637ced2e0132846d1af0d6` | `7139f69633f1dc2d12ceb44329fa4be766d395b48fb88dd8ccac767230e9b1c2` |

## Schedule and time rule
Two A100 jobs, walltime 10:00:00, tasks in series inside each job: card s42 = R, L20, W320, F3 with the s42 inits; card s43 = the same with the s43 inits. F3 is last, so a time shortfall drops it first. Planned time per task, from the measured B1024 A100 rate (6.3 step/s): R 1.4 h, L20 2.3 h, W320 2.0 h, F3 2.15 h, plus 300 s start-up each (estimates, plus or minus 30%; the total is about 8.2 h of the 10 h).
Before each task the driver (`run_width_500k_serial.py`) takes the pass time measured on R of the same card (median of at least 3 passes; the prior otherwise), scales it by the planned ratios, adds the start-up time, and starts the task only if that fits in the wall left minus 1800 s. If not, the task is recorded `SKIPPED_FOR_TIME`. A failed task (non-zero exit) is recorded `FAILED` and the next task still runs. SIGTERM or the wall limit pauses the running task at its atomic checkpoint (`PAUSED`); the same command resumes. Tasks that are not `COMPLETE` are tried again by the next submission, which keeps every rule here. The summary is `width_500k_summary_<card>.json` in the root, written atomically after each task (status, reason, start and end time, passes, measured samples/s, estimate).
A task is `COMPLETE` only when the runner wrote `training_complete.json`.

## Scoring (after retrieval; offline, local)
Retrieve each root's run folders to a new directory and check remote sha256, rows [0, 500,000), init sha, `arm_record.json` shape and zero non-finite values. Score `ema_model.pt` once per endpoint with `platforms/local/pcqm_tail_holdout_eval.py --run NAME=<ema_model.pt>:500000 --weights-key model --run-config <run_config.json> --initial-state <init file> --save-predictions <dir>`. The scorer reads the model shape from `arm_record.json` beside `run_config.json` and sets `MOLGAP_GPTRANS_{LAYERS,WIDTH,HEADS,FFN_RATIO}` itself; if `arm_record.json` is absent, set them by hand from the table. A wrong shape fails the strict weight load. No snapshot is picked, nothing is tuned on tail, no official valid or test is read.
Pre-scoring check, not a score: `run_config.json` must show `train_rows` 500000, `max_shards` 10 and `target_mean_eV` 5.413434769952119.

## Pre-registered verdict (code: [width_verdict.py](width_verdict.py), committed before any score)
Tail MAE in meV, equal weights, nothing fitted. gain(arm) = mean over the two seeds of [MAE(R, seed) - MAE(arm, seed)]. Paired bootstrap: per-row difference of absolute errors against R of the same seed, averaged over the two seeds; 2000 row resamples, `default_rng(42)`, same index sets for every comparison; upper bound = 99th percentile (one-sided 99%). Row resampling does not measure training noise.

- **Nomination (W320, F3).** NOMINATED iff gain >= 1.5 meV, AND both seeds individually have MAE(arm) < MAE(R), AND the bootstrap upper bound < 0. Otherwise NOT_NOMINATED. 500K only nominates; adoption needs a full-scale run that again clears 1.5 meV.
- **Calibration (L20 against R, same definitions).** gain <= 0.5 meV: NO_SIGNAL. gain >= 0.9 meV: PROXY_VALID (0.9 mirrors the full-scale depth-20 gain of 0.89 meV). 0.5 < gain < 0.9: UNCERTAIN. L20 never counts as a nomination.
- **What a NOT_NOMINATED arm means, by calibration.**

| Calibration | NOT_NOMINATED reads as |
|---|---|
| PROXY_VALID | No gain at 500K, and 500K does show capacity: the route is negative at 500K. |
| UNCERTAIN | Neither closed nor reopened. No full-scale run follows from 500K alone, and the route is not closed either. |
| NO_SIGNAL | 500K cannot see capacity effects. The zero result must not be used to close the width or FFN route. |
| UNAVAILABLE (L20 or R missing) | As NO_SIGNAL. |

A NOMINATED arm stands in every calibration; with NO_SIGNAL or UNCERTAIN it is flagged, not discarded.
- **Missing runs.** An arm needs R and the arm for both seeds. Any missing run makes that arm INCONCLUSIVE (never FAIL); its single-seed numbers are descriptive. A missing R makes every arm INCONCLUSIVE. A missing L20 makes the calibration UNAVAILABLE and nominations stand. A missing task (`SKIPPED_FOR_TIME`, `FAILED`, `PAUSED`) may be resubmitted under this protocol and is then read by the same rules.
- **Reported always:** the R s42 vs s43 difference, each arm's tail MAE per seed, and R against the earlier B1024 500K tail MAEs of the same inits (111.672 and 110.958 meV, `D:/molgap-artifacts/tail_holdout_eval_gptrans_b1024_500k_s42.json`, `_s43.json`): a descriptive regression check, not a gate (their recipe identity is not re-verified here).

## Stated limits
- Two seeds per arm, init-only noise (data order and dropout fixed): the 1.5 meV bar and the 99% bound do not cover full training noise.
- Row bootstrap measures tail-row noise only.
- 500K may not transfer to full scale; that is why a nomination does not adopt.
- Three arms compared on tail by fixed rules; no step selects among snapshots.
- Cost is in native units and measured after the run from `sessions.jsonl`; the A100 hours above are estimates, not measurements. Credits are not measured per job.

## Frozen identities
Runner `platforms/colab/pcqm_k1_b1024_production/train_full_b1024.py` sha256 `120d5571593c56647afe8f7c6f6060afeb7781ff8aea101fb9454fdac0ee5b37` (unchanged, LF bytes). Manifest sha256 `7d358a679d299bd5de6c5822e42dcef34b0e0b9fa9598c684709509c0b94021d`. File shas of the launcher, driver, PBS cards and the two changed `src` files are in [ims_submission_width_500k.md](ims_submission_width_500k.md) (computed on the final committed bytes).

## Records
Prospective trajectory: `prospective/width_500k/trajectory.json` (policy `pcqm-gptrans-width-500k-20261117`, [role_plan.json](role_plan.json), [decision.md](decision.md)). A trajectory authorises no training. Submission record: [ims_submission_width_500k.md](ims_submission_width_500k.md).
