# DeMol-CIN 500K serial chain on IMS (J1-J5 in one job), prepared 2026-11-21, not submitted

One A100 job runs five frozen recipes in series. Decision rules were frozen before any submit; this file adds
no science rule. Gates, budgets and admission follow `opus_ruling_20261121.md` (Opus, 2026-11-21), which amends ruling Q4.
Frozen sources, read from Git, not copied:
- Replication rules (a), (b) and the Opus rulings: `experiments/pcqm_demol_cin_s43_100k_ims_20261120/protocol.md`
  (branch `codex/exp/demol-cin-ims-replication-20261120`, base commit `ee0e791b9` of this branch).
- 500K transfer, hetero ensemble, J5 shift, cost and give-back rules: `experiments/pcqm_demol_cin_500k_ims_20261120/protocol.md`
  (branch `codex/exp/demol-cin-500k-ims-20261120`, commit `a7c5f7d74`; not merged here).
- Chain mechanics: `platforms/ims/v4_submit/README.md`, section "Serial chain".
- Plan: `D:/molgap-artifacts/ims_next_plan_20261120.md`, sections 3-4.

## Chain

Resources: queue H, 1 A100, `ncpus` 16, walltime 24:00:00. The launcher injects `--wall-s 86280` and `--memory-limit-bytes 30000000000`.
Reserve 1800 s. Allowed per item and in total: 84,480 s. The sum is 62,640 s.
A budget only decides whether an item may start. A started item runs until the wall and then pauses (exit 2; run the same submit command again).
J4's 7,200 s is a start threshold, not an estimate: J4 starts if 7,200 s remain, checkpoints every 512 steps, pauses at the wall,
and a resume job runs the rest. In the all-pass case expect J4 to finish in one resume job.

| Order | Item | Recipe | Seed | Init | Budget s | Gates (all must pass) |
|---|---|---|---|---|---|---|
| 1 | `j1-demol-100k-s43` DeMol-CIN 100K | `experiments/pcqm_demol_cin_s43_100k_ims_20261120/j1_demol_cin` | 43 | placeholder | 7200 | none |
| 2 | `j2-gptt-100k-s43` GPT-T pair-update-norm 100K | `.../j2_gptrans_pun` | 43 | `init/s43.pt` | 4140 | none |
| 3 | `j3-demol-500k-s42` DeMol-CIN 500K | `j3_demol_cin_s42` | 42 | placeholder | 28800 | dev delta J1-J2 <= -3.0 meV |
| 4 | `j5-gptt-500k-s44` GPT-T 500K, platform control | `j5_gptrans_pun_s44` | 42 | `init/s44.pt` | 15300 | same gate as J3 |
| 5 | `j4-demol-500k-s43` DeMol-CIN 500K | `j4_demol_cin_s43` | 43 | placeholder | 7200 | dev delta J3-J5 <= 0.0 meV (Opus ruling (ii)) |

- All DeMol items pass `--gpu-resident-cells` (compact inputs on the GPU, no host cell list).
- 100K items: IMS 100K view, train rows 0-99,999, dev rows 100,000-149,999. 500K items: IMS view `ogb-train-500k-scnet-v1`
  (manifest sha `6a53bb27`), train rows 0-499,999, dev rows 500,000-549,999 (topology shard 0010 is listed with role `development`).
- 500K recipes are the J1/J2 recipes with only the data, step, role and target fields rebound: 234,360 steps (60 passes), `checkpoint_every` 512,
  target mean/std of the 500K train rows, train role `be4b3c53` (equals the Cutlass 500K value), dev role `4299750a`
  (computed with the driver's own digest; equals the Kaggle value). Selection is `best-development-ema` for all five, so dev MAE is like for like.
  Batch 128, FP32, AdamW 1e-3, weight decay 0.05, EMA 0.999 as in the cohort. J5 is the Kaggle `init_s44` recipe (state sha `ff949650`;
  the spec binds the file sha `2415cdbf`) rebound to the IMS view and torch 2.7.1, seed 42 (the cohort batch order).
- J3 and J4 share the 500K data and differ in the seed only. The sampler uses its own generator (seed plus pass), so seed 42 repeats the cohort batch order.
- Files for the package: `init/s44.pt` is committed with `git add -f` (ruling Q6). `init/s43.pt` is already in the replication directory.

## What the gates do

- Dev gate (J3, J5): implements the "replicated" half of frozen rule (a): Delta43 = dev MAE(J1) - dev MAE(J2) <= -3.0 meV.
  The job cannot compute the confidence interval. After retrieval check that the paired row-bootstrap upper bound is below 0.
  If it is not, report the 500K results but flag them (rule (a) already says read them under their own rule and flag the conflict).
  A Delta43 between -3.0 and -1.5 is "inconclusive, no scale-up from 100K evidence"; the gate skips there too.
- No memory gate (Opus ruling (i), (iii)). Memory is an admission fact, checked once: the real 500K build peaked at 6.4 GiB private commit
  locally; with about 1 GiB of CUDA library pages that is about 7.4 GiB, under 0.5 x 30e9 bytes (14.0 GiB). The share is 16 cores x 1.875e9
  (RCCS manual, jsub page). The DeMol resident path holds one process (`FamilyLoaders.train()` returns the device stream when `data.train`
  has `train_stream`, `src/molgap/family_training_loaders.py`). The build peak equals the whole-run peak, so an OOM would fail J3 in about 5 minutes.
  J5 (GPT-T) forks one loader worker on Linux: its local 7.0 GiB could reach about 14 GiB in total, still under 30 GB.
- J4 gate (Opus ruling (ii)): J4 runs only if dev MAE(J3) - dev MAE(J5) <= 0.0 meV. It also needs J3 and J5 COMPLETE; J3 is gated by J1/J2.
  Limits: it reads dev, not tail; one seed each; J3 and J5 differ in architecture and init. It is a cost rule, not a verdict.
  The frozen 500K rule reads J3 and J4 together; with J4 skipped, the transfer rule caps at INCONCLUSIVE unless tail J3 <= 104.94 and its
  bootstrap upper < 0 against all cohort members. A NEGATIVE closure then needs J4 in a separate, separately decided job.
- A gate that cannot read its input skips (item not COMPLETE, no `selected_eval.json`): no evidence, no expensive run.
- A finished item is never redone. Gate skips are judged again only inside a resume job, and a resume job exists only after exit 2 (see Stop rules).

## Stop rules

- Delta43 > -3.0: J3, J5, J4 skip; the job ends after about 3-4 h. Do not fill the slot. Read rule (a) (close the lead above -1.5, inconclusive between).
- Only driver exit 2 (paused, time skip, not started) can resume: the launcher writes `paused`, and `resume_run` accepts only that
  (`platforms/ims/v4_submit/launcher.py`, `resume_run` and `run_stage`). Reconcile with `jobinfo -c`, then run the same submit command once.
  At most 2 resume jobs. Each must pass the maintenance guard; submit the last before 2026-12-06 03:00 JST (first Monday 2026-12-07 09:00
  minus 1.25 x 24 h). The guard ignores queue wait.
- Exit 0 (all COMPLETE or gate-skipped) and exit 1 (an item FAILED, nothing resumable) are final for this run id: the launcher writes
  `complete` or `error` and refuses a resume. A new run id means a new queue wait and fresh J1 and J2 runs.
- A FAILED item is retried (limit 2 attempts) only inside an exit-2 resume. Do not change a recipe to rescue a run. No weight-decay or schedule grids.
- If J3 FAILED, J5 still runs (up to its 15,300 s budget, about 4.25 h), J4 is SKIPPED_BY_GATE, and the job exits 1 with nothing to resume.
  J5 alone still measures the platform shift.
- Exit 3: the start-of-job check rejected an input; read `early_exit.json`, fix, and re-review. No blind resubmit.
- A quiet or unknown job is `UNKNOWN`, not failed. Never submit a successor because local status is stale.

## After retrieval

1. `jobinfo -c`. Retrieve each item's `training/` files with remote sha256 into `D:\molgap-artifacts\retrieved\demol500k-chain-20261121\<item>\`.
2. Read `serial_summary.json`: statuses, reasons, per-item wall, `peak_rss_bytes` (the first Linux peak), gate texts. Take the A100 step rate from `train_log.jsonl`.
   Log them as measured cost; everything else stays estimated.
3. Check each `completion_manifest.json` and `run_provenance.json` (source commit, no protected role read) and that `selected_eval.json` equals the `train_log.jsonl` dev MAE.
4. Score offline with `platforms/local/pcqm_tail_holdout_eval.py`, DeMol with this checkout's `src` on `PYTHONPATH`: dev rows first with `--expect-dev-mae` from `selected_eval.json`,
   then the tail once per selected checkpoint (J1, J2, and each 500K item that ran). Do not score Cutlass/GRIT 500K tails. No selection on the tail. Official valid and test stay sealed.
5. Rule (a): dev delta with the paired row bootstrap (2000 resamples, seed 42), two-seed mean with s42 for the report only.
6. Rule (b): P from per-row dev predictions of G42, G43, D42, D43. Compute P500 on the tail rows if J3 and J4 both ran.
7. 500K transfer rule, J5 shift rule, cost readout (DeMol train wall over J5 train wall) and give-back readout, exactly as in the frozen protocol.
   Single-model verdicts across platforms are not allowed (Kaggle T4 torch 2.11 against IMS A100 torch 2.7.1); J5 is the bridge.
8. Write `decision.md` with the date, then terminal attribution beside it, then close RML once. Route the branch per BRANCHES.md.

## Measured locally (RTX 5060, Windows, single process, `--training-workers 0`; not an A100 or Linux number)

See `local_rehearsal_record.json` for each check and `Local checks` below.

- Real 500K compact build (train 500,000 + dev 50,000 rows), `platforms/local/demol_cin_resident_check.py mem --variant new`:
  build 311.8 s single thread; host peak working set 4,064 MiB (process-tree sum equals the single process); peak private commit 6,401 MiB;
  compact store 882.7 MiB (1,682.8 bytes per molecule); GPU 930.5 MiB resident after model and store, 1,507.3 MiB peak over 50 steps.
  The old linear extrapolation was about 7.3 GB host and about 0.9 GB GPU; the host figure was 1.8 times too high.
- The 100K run (150K rows) peaked at about 2.0 GiB, so host memory grew by 2.0 times for 3.7 times the rows. Not measured: Linux `ru_maxrss`.
- Rate, one set of numbers (Opus ruling, "Step rate"): local 5060 at most 0.099 s per step for DeMol 500K, derived from the J3 one-pass
  rehearsal (699 s wall - 312 s build, over 3,906 steps, with one dev pass and start-up inside). The rehearsal operator reports about
  0.087 s per step steady state; the record holds no trace of it. The 0.136 s figure above is a 50-step check with warm-up; 0.141 (plan) is not a steady rate either.
  The A100 rate is unmeasured: frozen estimate 60-110 ms; DeMol is launch-bound (about 6,400 kernels per update) and may be no faster.
  So J3 and J4 each need about 4-9 h for 234,360 steps; in the all-pass case J4 likely pauses and one resume job finishes it.

## Local checks

Real flow on the real spec, all with `--training-workers 0`, one GPU task at a time, under a handle and working-set guard. The run reused the
rehearsal helpers (`platforms/ims/family_serial_rehearsal.py`) through `local_rehearsal.py`. Differences from IMS: no `--accelerator A100`, no lustre `--ogb-zip`,
one-pass copies of the recipes in the chain part, and a hard-linked local mirror of the fixed root (topology shards 0000-0010 extracted from the archive; each file's
sha256 and size equal the manifest; the 500K manifest was regenerated by the builder's own rules and its sha equals the registry value `6a53bb27`; no dataset was rebuilt).
Results: `local_rehearsal_record.json`. The `early` and `launcher` parts were rerun on 2026-11-21 on the amended spec (fresh work dir);
the chain parts (A-D) ran on the earlier spec (memory gates present, J4 line -1.5, J4 budget 28,800) and were not rerun.

Outcomes (details in the record; every check PASS):
- Real spec and real 100K/500K mirror data: spec and input checks return no problem with `ogb` blocked.
- Exit 3 with a named problem and no child, for 14 broken inputs: recipe sha, J2 and J5 init sha, train role identity, final-ema gate item,
  mixed 100K/500K dev roles, budget above 84,480, driver flag in `extra_args`, gate naming a later item, memory limit 0, 500K manifest missing,
  dev shard 0010 missing, dev shard 0010 wrong size, train shard 0007 wrong size.
- Chain (one-pass copies, same names, gates and flags as the real spec): kill during J1 then resume from step 128; real dev gate (-3.0) skips J3, J5 and J4;
  failed probe retried once then left at the attempt limit; time probe skipped for time; memory gate shut: J3 SKIPPED_FOR_MEMORY, J5 runs, J4 skipped because J3 is not COMPLETE;
  all gates open: J3, J5 and J4 (GPU-resident DeMol 500K) complete; J4 gate forced shut skips with the measured delta in the reason; 500K DeMol item paused by its own wall and resumed.
- Launcher: real run_spec validates (mismatched `--cpus`, 25 h, typed wall refused); `train.pbs` passes `bash -n` with ncpus 16, ngpus 1, 24:00:00, no `mem=`;
  args expand to `--wall-s 86280` and `--memory-limit-bytes 30000000000`; the worker run recorded the same two numbers.
- Guard: peak handles 1,785 (limit 20,000), peak working set 4.4 GiB (limit 22 GiB); no trip. Local one-pass 500K: J3 3.98 GiB, J5 7.01 GiB; J1 100K 2.25 GiB.
- Weaker than real: one pass (781 or 3,906 steps) instead of 60 passes, so the gate deltas are meaningless; the 500K pause came at step 1 (data build exceeded the 300 s wall), so it proves pause and resume, not a mid-run checkpoint at 500K (mid-run resume was shown at 100K, step 128).

## Not verified here

- Linux memory peak (`ru_maxrss`), A100 accelerator check, A100 step rates, the real queue, SIGTERM handling (Windows has no SIGTERM; the driver's own wall pause was tested instead).
- The scheduler's memory enforcement and the H versus PN queue name (README of v4_submit lists both as unverified).
- Submit-day preconditions (Opus ruling, Q2 and Q3): `jobinfo -c` and `showlim -c` under REMOTE_HANDOFF. Q2: this chain counts as one job;
  non-terminal MolGap chains + 1 <= 4; jobs 2077090-2077094 must be gone (if live or with outputs, stop and reconcile). Q3: remaining minus
  hard line >= 3 x 24 h x 73.1 points per A100-h (about 5,263 points). Then an Opus review of the final commit.
- Ruling Q4 asked for an IMS-measured RAM fact that cannot exist (30 GB at 16 cores). Opus amended it on the local 500K measurement; see `opus_ruling_20261121.md`.
- To do, not on this branch: `platforms/ims/v4_submit/README.md` lines 98-100 (cgroup and MemAvailable probe the driver no longer uses) and
  121-128 are stale. Fix them on the infra branch.
- Maintenance guard: refuses a submit at or after Sunday 2026-12-06 09:00 JST, or when now + 1.25 x 24 h reaches Monday 2026-12-07 09:00 JST,
  so the last submit is before 2026-12-06 03:00 JST. Queue wait can still push the run into the window; check the expected start.
