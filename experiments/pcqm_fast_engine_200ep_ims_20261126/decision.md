# Fast engine, 200 passes at 500K: longer training did not help the tail

2026-11-28, Track B, infrastructure work on the fast engine. IMS run
`fastpn-200ep-20261126` (job 2176763, one item `fast-500k-s42-e200`) finished
COMPLETE. Rules: [protocol.md](protocol.md), "Read rule". The 500K 60-pass
closure is in `../pcqm_fast_engine_500k_ims_20261125/decision.md` (merged into
this branch). Closure built with `rml legacy-terminal` and `terminal-pipeline`.

## Verdict

**NEGATIVE_UNDER_CONTRACT.** The protocol declared "longer helps" only if tail
MAE is at most 106.02 meV. Tail MAE is 107.761 meV: the gate is not met. Per the
protocol the reading is "no detectable change". effect, ci and source are
null: the two score files saved no per-row predictions
(`per_row_predictions_saved: false`), so no paired comparison is possible, and
no tail row may be read again. failure_mode is null. No next action is allowed.

## Scope (fixed)

This holds only for: 500K training rows, batch size 128, seed 42, the fast
engine, and a cosine schedule re-spread over 200 passes (warmup 4 passes).
It is not extended to the full 3.37M set. It is not extended to 20 layers; the
20-layer question is answered separately by IMS jobs 2187198 and 2187199.

## Accuracy (descriptive, one run each)

| Quantity | 60 passes | 200 passes | Difference |
|---|---:|---:|---:|
| Tail MAE, final EMA, FP32 scorer (meV) | 107.522 | 107.761 | +0.239 (worse) |
| Dev MAE rows 500,000..549,999, final EMA (meV) | 101.972 | 100.391 | -1.581 (better) |

Dev and tail point in opposite directions. Dev is report-only by protocol and
selected nothing; the tail value decides the gate. The tail difference (+0.24
meV) lies well inside the seven-baseline spread (1.08 meV), and the 200-pass
value 107.761 also lies inside the baseline range 107.273 to 108.357. Gate
threshold: 106.02 meV. Not met by 1.74 meV.

The two numbers are from two separate jobs and two separate scoring files; no
interval is claimed. A dev gain with no tail gain is consistent with fitting the
50,000 dev rows more closely at the long horizon, but this record does not test
that, and none is claimed.

## Run facts

- 781,200 optimizer steps, 200 passes. `scaler_skips` 356 (0.046% of steps);
  `nonfinite_steps` 0.
- `pass_s`: pass 1 411.0 s; mean of passes 3..60 189.9 s; mean of passes 3..200
  186.0 s (range 180.1 to 206.6). The in-job stop (three passes above 289 s) never
  fired. Sum of `pass_s` 37,439 s; final `elapsed_s` 38,186 s; item wall
  38,214.3 s (10.615 h); one attempt, no resume. Peak RSS 2.98 GB.
- Source commit `1006873691ae8120fea0ead03f3a6bf3db42bc6e`. Spec sha256
  `be38d65f4b7fd429ffb83b5845c3678a8334208ddd62c87bd46f03e48f2a583c`. Recipe
  sha256 `c26ca850dbc85d373beb7bc0642a239b791d03f47a1154a9f6862c70db624e51`.
- `compile_used: true` only says compile was enabled.

## Role use (structural evidence)

- `official_validation`, `test_dev`, `test_challenge`: **untouched**.
  `run_provenance.json` records `official_valid_read`, `test_read` and
  `tail_holdout_v1_read` false for training; the tail score file records
  `protected_roles_read` all false.
- `tail_holdout_v1`: **consumed**, one offline scoring of the final EMA
  (`metric_computed`); no selection used it.
- `dev_500k`: **consumed**, report only; selection is `final-ema`
  (`selection_used` false).

## Custody

- `selected_model.pt` sha256
  `b76c5457c1f946b23d9524f021666e18e2e50e81e6ba8bdce34063ba0e64d9ab`, at
  `D:/molgap-artifacts/fastpn-200ep-20261126/retrieved/training/fast-500k-s42-e200/selected_model.pt`
  (not in the repo). Last checkpoint sha256
  `fdc6c97061972bf1fdc0a78b696d6e3cb6f5f0342456405795a1ab8c627e5521`.
- Tail score file: `D:/molgap-artifacts/tail_eval_fast200_20261126/fast_200ep_s42.json`
  (copy in `evidence/`). Scorer sha256 `874c25ee...750f123`.
- Cost: one measured A100 wall figure (10.615 h); CPU and queue hours not measured.

## Not authorised

Further tail scores, official-valid and test reads, longer or larger runs on
this recipe, and any adoption. Reopen only with new explicit user authority.
