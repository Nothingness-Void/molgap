# Fast engine, 200 passes on IMS A100: protocol

Frozen 2026-11-26 JST before any rehearsal or training. Track B, branch `codex/exp/fast-engine-200ep-ims-20261126`
(from `codex/exp/fast-engine-500k-ims-20261125`, HEAD 1e7e1472d). This file submits nothing. The main session submits after an
Opus pre-submit review.

## Question

Does a 200-pass cosine fast GPT-T beat the 60-pass fast GPT-T (tail 107.52 meV, seed 42)?

## Same as the 60-pass fast run (`fastpn-500k-20261125`, item `fast-500k-s42`)

Everything in `recipe.json` except the pass count: model, init `s44.pt`, seed 42, train rows 0..499,999, batch 128 (drop last),
lr 1e-3, 4 warmup passes, cosine to 1e-6, AdamW wd 0.05, EMA 0.999, policy `fast_amp_nondeterministic`, selection `final-ema`,
runtime `ims-a100-torch271-v1`. Changed: 200 passes (781,200 steps, 99,993,600 sample presentations); the cosine now spans 200
passes. The warmup stays 4 passes. Item `fast-500k-s42-e200`, run `fastpn-200ep-20261126`.

## Time

The 60-pass run took 192.7 s per pass (mean of passes 3..60, A100), 12,115 s in all. Expect about 40,500 s for 200 passes.
`budget_s` 52,000 (28% margin). Wall time 15:00:00 (the launcher caps it at 24 h): 54,000 - 120 - 1,800 reserve = 52,080 >= 52,000.
If the wall ends first the job pauses (exit 2); run the same `submit_from_windows.py` command to resume.

## In-job stops (driver `fast_recipe_run.py`, checked after each pass)

Non-finite loss or dev MAE; more than 1% GradScaler skips in one pass; three passes in a row above 289 s (1.5 x 192.7 s, pass 1
excluded). Recipe, init and target-stat hash mismatches are refused at job start as before.

## Read rule (declared before the run)

Final EMA only. Tail read once with `platforms/local/pcqm_tail_holdout_eval.py` (FP32). Verdict "longer helps" only if tail
<= 106.02 meV (at least 1.5 meV below 107.52, and beyond the 1.08 meV spread of the 7 FP32 baselines). Otherwise "no detectable
change". Dev curves are descriptive only. No snapshot selection, no official valid, no test.
