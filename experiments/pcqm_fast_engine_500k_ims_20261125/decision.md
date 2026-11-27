# Fast engine on 500K GPT-T (IMS A100): no detectable accuracy change, 2.42x faster per pass

2026-11-28, Track B, infrastructure (speed) work. IMS run `fastpn-500k-20261125`
(job 2161635, one item `fast-500k-s42`) finished COMPLETE. Rules: [protocol.md](protocol.md)
("No fixed gate: read against noise"). The Kaggle Wave-1 thresholds
(2.0x speed, 1.5 meV) belong to a different job and are not used here. Closure
built with `rml legacy-terminal` and `terminal-pipeline`.

## Verdict

**INFRASTRUCTURE_ONLY.** Accuracy: no detectable change. Speed: 2.42x per pass,
one run. Nothing is adopted by this job. effect, ci and failure mode are null:
there is no same-job reference arm and no paired per-row file.

## Accuracy (descriptive, one run)

| Quantity | Value |
|---|---:|
| Tail MAE, final EMA, FP32 offline scorer (`evidence/tail_score_fast_500k_s42.json`) | 107.522 meV |
| Seven same-contract FP32 tail baselines | 107.273 to 108.357 meV (range 1.084, mean 107.86) |
| Dev MAE rows 500,000..549,999, final EMA (offline scorer / engine log) | 101.972 / 101.973 meV |

Tail 107.522 lies inside the seven baselines: **no detectable change**.

The protocol's second condition (FP32, non-deterministic kernels, buckets kept)
is triggered only if the tail value falls outside the range. It did not, so it
was not run. The consequence: this run changes AMP FP16, non-deterministic
kernels, batch order, fused AdamW, padding and compile together, and one run
cannot split their effects. "No detectable change" holds for the bundle, not
for AMP alone and not for non-determinism alone.

Dev is 1.26 meV worse than the baseline `ctl-500k-s42` (100.713 meV, P1 job
2146337). This is a cross-job descriptive value, smaller than the 500K dev
init range of 1.332 meV; the protocol makes dev report-only. It is not
a ranking and not a gate.

## Speed

Recomputed from `evidence/train_log.jsonl` (`pass_s`, passes 3..60, 58 passes):

- Mean `pass_s` 192.71 s. Baseline `ctl-500k-s42` 466.6 s per pass (passes 3..60,
  FP32 deterministic, same A100 model). Ratio 466.6 / 192.71 = **2.42x**.
- Pass 1: 438.46 s (includes store build and compile). Pass 2: 208.38 s.
  Last pass: 190.87 s.
- Sum of `pass_s` over 60 passes: 11,824 s. Final `elapsed_s` 12,088 s (includes
  evaluation). Item wall 12,114.6 s (3.365 h); one attempt, no resume.
- Peak RSS 3.00 GB.

Not comparable between job and baseline: precision, determinism, batch order,
padding, optimizer rounding, compile, node and queue time, and evaluation
precision (engine AMP, scorer FP32). The 2.42x is a ratio of two separate jobs.

## What the speed is not credited to

`run_provenance.json` records `compile_requested: default` and
`compile_used: true`. That says compile was switched on. It does not say it
worked or helped. The job's stdout and stderr were not retrieved; only the
structured outputs were. The retrieved files hold no dynamo warning, but they
could not hold one, so the absence of warnings is **unchecked**. The engine
changes ship as a bundle (AMP, GPU-resident batches, fused AdamW, static
buckets, compile) and no ablation separates them. The speed gain is credited to
"the bundle", not to torch.compile.

## Other facts

- `scaler_skips` 101 over 234,360 optimizer steps (0.043%); `nonfinite_steps` 0.
- Source commit `1e7e1472d9bb7143ff246015ebe79df4b637b1aa`; trajectory froze
  `114e3affc4e5205ebd0cf8903dc2af3629295073` at planning. Spec sha256
  `8f9a7b2d7e93a05b270d2f4d17b377d6b0e6edb0d45e2b9050e0c92ed6e75962`.
- `trajectory.json` still shows `chosen_action: LOCAL_REHEARSAL` (stale). It does
  not affect closure: action A002, run `fastpn-500k-20261125`, as frozen in the
  trajectory.

## Role use (structural evidence)

- `official_validation`, `test_dev`, `test_challenge`: **untouched**.
  `run_provenance.json` records `official_valid_read` and `test_read` false;
  the tail score file records `protected_roles_read` all false.
- `tail_holdout_v1`: **consumed**, one offline scoring of the final EMA
  (`metric_computed`). `run_provenance.json` records `tail_holdout_v1_read` false
  for training. No selection used it.
- `dev_500k`: **consumed**, report only. The engine logs the EMA dev MAE every
  pass; selection is `final-ema`, so dev never chose a checkpoint
  (`selection_used` false).

## Custody

- Checkpoint `selected_model.pt` sha256
  `6dbdd7cebf5a2a0635a9aecc9a34a041b4806e68d0113997be0410d42dec0c72`, at
  `D:/molgap-artifacts/ims_retrieved_20261125/fastpn/training/fast-500k-s42/`
  (not in the repo). Last checkpoint sha256
  `8244bc07322d7a774122ae053b38901a5b6311d9ffac24c07f4215fd8405bc56`.
- Tail score file: `D:/molgap-artifacts/tail_eval_fast_engine_500k_20261125/fast_500k_s42.json`
  (copy in `evidence/`). Scorer sha256 `874c25ee...750f123`. Per-row predictions
  were not saved.
- Cost: one measured A100 wall figure (3.365 h); CPU and queue hours not measured.

## Not authorised

Adopting the engine, a larger or longer run with it, a second condition, any
further tail score, official-valid and test reads. Reopen only with new explicit
user authority.
