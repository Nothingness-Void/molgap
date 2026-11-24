# Fast engine 500K on IMS A100: protocol

Frozen 2026-11-25 JST before any rehearsal or training. Track B, branch `codex/exp/fast-engine-500k-ims-20261125`
(from `codex/exp/fast-engine-200ep-3d-kaggle-20261124`). This file does not submit anything. The main session submits
after an Opus pre-submit review.

## Question

Does the fast engine train the same 500K GPT-T BondLocal + PairNorm recipe to a tail MAE inside the existing noise range, and
faster per pass? One job, one item (`fast-500k-s42`). No fixed gate: read the result against noise.

## Same as the baseline `ctl-500k-s42` (P1 job 2146337, run `gptdftp1-chain-20261123`)

Model (5,871,201 params), init `s44.pt` (file sha 2415cdbf..., state sha ff949650...), seed 42, train rows 0..499,999, 60 passes,
234,360 steps, batch 128 (drop last), lr 1e-3, 4 warmup passes then cosine to 1e-6 (the family `ScheduleSpec`), AdamW wd 0.05,
betas (0.9, 0.999), eps 1e-8, clip 1.0, EMA 0.999, dropout 0.1, drop-path 0.1, selection `final-ema`, runtime
`ims-a100-torch271-v1`. Target mean 5.413434769952119, std 1.176728647311213: the recipe's values. The driver recomputes them from
the decoded train labels (abs 1e-12) and refuses a mismatch.

## Changed (the named policy `fast_amp_nondeterministic`)

AMP FP16 with a GradScaler (`scaler_skips` logged), non-deterministic kernels, fused AdamW, GPU-resident batches with static
20-atom and 256-edge buckets, shortest paths computed once on the GPU, one pinned `torch.compile(mode="default")` when Triton is
importable (eager otherwise; `run_provenance.json` records which). Batch order is `randperm(seed + pass)`, not the baseline's
`torch-pass-randperm` stream. The policy is declared in the run contract, the serial item and the recipe; all three must agree.
Pad-to-20 holds: the largest molecule in one checked 500K shard is 18 atoms, so shapes vary only by edge bucket.

## Metrics

- Accuracy, primary: the final EMA on `tail_holdout_v1`, scored once with `platforms/local/pcqm_tail_holdout_eval.py`
  (`--recipe` the baseline `ctl_s42/recipe.json` of the P1 branch, `--initial-state` s44.pt, `--weights-key model`, FP32). The
  baseline recipe is used on purpose: the scorer rebuilds the family model, and the fast recipe is not a family recipe.
- Reference: seven same-contract FP32 tail values, 107.273 to 108.357 meV, mean 107.86, range 1.084 (leaderboard
  `D:/molgap-artifacts/tail_holdout_master_leaderboard.json`). 500K init noise on dev: range 1.332 meV, SD 0.520 meV
  (`experiments/pcqm_gptrans_pairnorm_init_noise_500k/decision.md`). `ctl-500k-s42` has no tail score and none is read.
- Secondary, report only: dev MAE (rows 500,000..549,999) of the EMA every pass (`ema_dev_mae_eV`) and of the live weights every
  10th pass (`dev_mae_eV`). Baseline `ctl-500k-s42` final dev: 100.713 meV.
- Speed: mean `pass_s` over passes 3..60 (training seconds of a pass, evaluation excluded, checkpoint writes included) against the
  baseline 466.6 s/pass (passes 3..60 of `retrieved_job1/ctl-500k-s42/train_log.jsonl`, same A100 model, FP32). Also report
  `elapsed_s` differences and total wall. `elapsed_s` counts active seconds across resumes; pass 1 includes store build and compile.

## Reading

An accuracy gap inside the reference range is "no detectable change"; outside it is a flag, not a verdict (one run). One run cannot
split non-determinism from AMP precision; a second condition (FP32, non-deterministic, buckets kept) would, and is added only if the
tail value lands outside the range. Speed ratio = 466.6 / fast `pass_s`.
Not comparable between job and baseline: precision, determinism, batch order, padding, optimizer rounding, compile, node and queue
time, evaluation precision (engine AMP, scorer FP32).

## Not done

No batch-size 256/512 probe: the run contract pins `physical_batch_per_device: 128` for the whole job, so such an item would
misstate it. The job reads no official valid, test or tail row.

## Local rehearsal (2026-11-25, RTX 5060, eager: no Triton on Windows)

`family_serial_rehearsal.py spec` on the committed tree (HEAD 073303964), record in `local_rehearsal_record.json`: spec checks,
start-of-job check on the real spec, 20 real steps, kill after a checkpoint, resume at step 128, one-pass chain COMPLETE. One pass
took 280 s of training (about 72 ms/step; the 5060 FP32 baseline is about 330 s per pass, but one pass is not a speed claim:
one pass, eager), evaluation 11 s. The saved `selected_model.pt` loads strictly into the family model and gives dev MAE 0.227502 eV in FP32,
against 0.227494 eV logged by the engine in AMP. The inverse key layout reproduces the init state sha ff949650... exactly.
Not rehearsed: `torch.compile` (needs Triton), the Linux memory probes, the A100 name check.
