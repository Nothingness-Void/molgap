# DeMol-CIN 500K serial chain on IMS (J1-J5 in one job), prepared 2026-11-21, not submitted

One A100 job runs five frozen recipes in series. Decision rules were frozen before any submit; this file adds
no science rule except the one marked NEW RULE. Frozen sources, read from Git, not copied:
- Replication rules (a), (b) and the Opus rulings: `experiments/pcqm_demol_cin_s43_100k_ims_20261120/protocol.md`
  (branch `codex/exp/demol-cin-ims-replication-20261120`, base commit `ee0e791b9` of this branch).
- 500K transfer, hetero ensemble, J5 shift, cost and give-back rules: `experiments/pcqm_demol_cin_500k_ims_20261120/protocol.md`
  (branch `codex/exp/demol-cin-500k-ims-20261120`, commit `a7c5f7d74`; not merged here).
- Chain mechanics: `platforms/ims/v4_submit/README.md`, section "Serial chain".
- Plan: `D:/molgap-artifacts/ims_next_plan_20261120.md`, sections 3-4.

## Chain

Resources: queue H, 1 A100, `ncpus` 16, walltime 24:00:00. The launcher injects `--wall-s 86280` and `--memory-limit-bytes 30000000000`.
Reserve 1800 s. Allowed per item and in total: 84,480 s. Budgets are the planner's mid estimates; the sum is 84,240 s.
A budget only decides whether an item may start. A started item runs until the wall and then pauses (exit 2; run the same submit command again).

| Order | Item | Recipe | Seed | Init | Budget s | Gates (all must pass) |
|---|---|---|---|---|---|---|
| 1 | `j1-demol-100k-s43` DeMol-CIN 100K | `experiments/pcqm_demol_cin_s43_100k_ims_20261120/j1_demol_cin` | 43 | placeholder | 7200 | none |
| 2 | `j2-gptt-100k-s43` GPT-T pair-update-norm 100K | `.../j2_gptrans_pun` | 43 | `init/s43.pt` | 4140 | none |
| 3 | `j3-demol-500k-s42` DeMol-CIN 500K | `j3_demol_cin_s42` | 42 | placeholder | 28800 | dev delta J1-J2 <= -3.0 meV; memory |
| 4 | `j5-gptt-500k-s44` GPT-T 500K, platform control | `j5_gptrans_pun_s44` | 42 | `init/s44.pt` | 15300 | same two gates as J3 |
| 5 | `j4-demol-500k-s43` DeMol-CIN 500K | `j4_demol_cin_s43` | 43 | placeholder | 28800 | NEW RULE: dev delta J3-J5 <= -1.5 meV |

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
- Memory gate (J3, J5): skip if peak(J1) x 5 x 1.3 > 0.8 x 30e9 bytes (24e9 bytes, 22.35 GiB). That allows a J1 peak up to 3.44 GiB.
  - Limit: 16 cores x 1.875e9 = 30e9 bytes (RCCS manual, jsub page; see the v4_submit README).
  - Fraction 0.8, not 0.5: the DeMol resident path has no loader worker. `FamilyLoaders.train()` returns the device stream when
    `data.train` has `train_stream` (`src/molgap/family_training_loaders.py`, branch on `hasattr(data.train, "train_stream")`); the `DataLoader` built
    with one worker is never iterated; the dev set uses `sequential_stream`; calibration (the only other loader) needs `clean_bn`, which these recipes lack.
    So the job holds one process, and the driver measures that process with `ru_maxrss`. Fork workers would need 0.5. FLAGGED for Opus.
  - Local measurement (Windows, one process, 0 workers): see "Measured locally". The real 500K peak is far below the x5 x1.3 proxy.
    At 0.5 the gate would wrongly skip J3 on local numbers: J1 peak 2.25 GiB x 6.5 = 14.6 GiB is above 0.5 x 27.94 = 13.97 GiB, while J3 really peaked at 3.98-4.06 GiB.
  - J5 carries the memory gate only because J5 has no value without J3 (frozen text: J5 depends on J3).
- NEW RULE, pending Opus review (planner proposal in the plan, section 4; not in the frozen protocol): J4 runs only if dev MAE(J3) - dev MAE(J5) <= -1.5 meV.
  Reason: it saves 6.5-9.5 A100 hours when the first 500K seed shows no dev edge over the same-platform control. J4 has no other gate of its own:
  it needs J3 and J5 COMPLETE, and J3 is gated by J1/J2. J4 has no memory gate (J3 COMPLETE already proves the fit).
  The frozen 500K rule still reads J3 and J4 together; with J4 skipped, the transfer rule caps at INCONCLUSIVE unless tail J3 <= 104.94 and its bootstrap upper < 0 against all cohort members.
- A gate that cannot read its input skips (item not COMPLETE, no `selected_eval.json`, no measured peak): no evidence, no expensive run.
- A skipped item is not final: rerunning the same command re-evaluates its gate. A finished item is never redone.

## Stop rules

- Delta43 > -3.0: J3, J5, J4 skip; the job ends after about 3-4 h. Do not fill the slot. Read rule (a) (close the lead above -1.5, inconclusive between).
- A FAILED item is retried once on rerun (limit 2 attempts), then left. Do not change a recipe to rescue a run. No weight-decay or schedule grids.
- Exit 2 (paused, time skip, not started): reconcile with the scheduler (`jobinfo -c`), then run the same submit command once. J4 may end `SKIPPED_FOR_TIME`; one more queue wait is accepted.
- Exit 1: an item failed and nothing can resume. Exit 3: the start-of-job check rejected an input; read `early_exit.json`, fix, and re-review. No blind resubmit.
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
  compact store 882.7 MiB (1,682.8 bytes per molecule); GPU 930.5 MiB resident after model and store, 1,507.3 MiB peak over 50 steps; 0.136 s per step.
  The old linear extrapolation was about 7.3 GB host and about 0.9 GB GPU; the host figure was 1.8 times too high.
- The 100K run (150K rows) peaked at about 2.0 GiB, so host memory grew by 2.0 times for 3.7 times the rows. The gate proxy (x5 x1.3 = x6.5) is about 3 times too cautious on this evidence.
  Not measured: the Linux `ru_maxrss` of J1. A false skip happens only if that peak exceeds 3.44 GiB.
- Rate: 0.136 s per step on the 5060 gives 8.9 h for 234,360 steps. The A100 rate is unmeasured (DeMol is launch-bound, about 6,400 kernels per update).
  If the A100 is no faster, J3 and J4 each use about 8.9 h and the all-pass case needs about 25 h: J4 pauses or skips for time, and one resume job is needed.

## Local checks

Real flow on the real spec, all with `--training-workers 0`, one GPU task at a time, under a handle and working-set guard. The run reused the
rehearsal helpers (`platforms/ims/family_serial_rehearsal.py`) through `local_rehearsal.py`. Differences from IMS: no `--accelerator A100`, no lustre `--ogb-zip`,
one-pass copies of the recipes in the chain part, and a hard-linked local mirror of the fixed root (topology shards 0000-0010 extracted from the archive; each file's
sha256 and size equal the manifest; the 500K manifest was regenerated by the builder's own rules and its sha equals the registry value `6a53bb27`; no dataset was rebuilt).
Results: `local_rehearsal_record.json`.

Outcomes (details in the record; every check PASS):
- Real spec and real 100K/500K mirror data: spec and input checks return no problem with `ogb` blocked.
- Exit 3 with a named problem and no child, for 13 broken inputs: recipe sha, J2 and J5 init sha, train role identity, final-ema gate item,
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
- Whether jobs 2077090-2077094 are really gone, the live account cap (ruling Q2), credits (ruling Q3), and Cutlass/GRIT state: all need `jobinfo -c` and `showlim -c` at submit time.
- Ruling Q4 asked for an IMS-measured RAM fact. None exists. The in-job memory gate replaces it; ask Opus to accept that.
- `platforms/ims/v4_submit/README.md` lines 98-100 still describe the cgroup and MemAvailable probe that the driver no longer uses (the driver takes `--memory-limit-bytes`). The text is stale, the code is right.
- Maintenance guard: a first-Monday window starts Sunday 09:00 JST; the next Monday is 2026-12-07, so the window opens 2026-12-06 09:00 JST. The guard also needs now + 1.25 x 24 h before that time. Queue wait can still push the run into the window; check the expected start.
