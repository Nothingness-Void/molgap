# C4: full-scale GPT-T, weight decay only (frozen 2026-11-23, before any result)

Authority: user, 2026-11-23 ("all approved", unattended run). Source of the arm: the court ruling (`final_ruling.md`, item 7: "C4 WD-only full arm, lambda about 0.022", last, after C3). C3 is done (PASS), so C4 is released. Branch `codex/exp/gptrans-full-wd-c4-20261123`, from `codex/exp/gptrans-full-levers-20261115` (33ab45716).

## Question
Does lowering AdamW weight decay from 0.05 to 0.022, and nothing else, lower full-scale tail MAE of GPT-T B1024?

## Where 0.022 comes from
The court files give the rule, "AdamW timescale matched in passes from 500K B128 to full B1024", but not the sum. I redid it: decay per pass = lr x wd x steps per pass.
- 500K B128: 1e-3 x 0.05 x 3,906 = 0.195.
- Full B1024: lr 2.8284e-3, 3,173 steps per pass. Same 0.195 needs wd = 0.195 / (2.8284e-3 x 3,173) = 0.0218, which rounds to 0.022.
The reference recipe keeps wd 0.05, so its decay per pass is 0.449, 2.3 times larger than the 500K recipe. I read lambda as the AdamW `weight_decay` value itself.

## Arm
Same as the s43 B1024 reference (rows [0, 3,250,000), 60 passes, warmup 4, cosine, B1024, lr 2.8284e-3, EMA 0.992028, s43 init file sha `2ccd1eb7...5f3c`, state sha `c95da412...5f5b`, endpoint final-EMA), except `weight_decay = 0.022`. Runner `train_full_b1024.py` stays byte-identical (sha `120d5571...5b37`); the launcher `run_gptrans_c4.py` sets the recipe field after the recipe is built. Code tree on IMS is a copy of the lever-D root (12 layers, env default), which gives the same numbers as the reference tree at 12 layers.

IMS root `/lustre/home/users/sm2/chou/molgap-gptrans-c4-wd022-s43-20261123`, run dir `gptrans-b1024-wd022-minus-tail-s43-20261123`, one A100, 12 h walltime, expected about 8.5 h (references measured 8.4-8.6 h). If cut by walltime, resubmit the same command.

## Reference and scoring
Reference: GPT-T B1024 s43, tail MAE 87.846 meV (full-minus-tail decision). Same init, same data order and dropout seed, so the pair isolates weight decay. Score once: tail_holdout_v1 (rows 3,250,000 to 3,378,605), final-EMA weights, existing tail evaluator. 2000-sample paired bootstrap over tail rows, 95% interval of (C4 - reference). Official valid/test, sealed 20K and the production registry stay untouched.

## Gates (meV, negative is better)
- Delta <= -1.0 and bootstrap upper bound < 0: real effect. Delta <= -1.5: worth adopting for the next full-scale recipe.
- -1.0 < Delta <= -0.5: inconclusive, no adoption.
- Delta > -0.5: no effect, closed.

## Caveats stated now
- One init and one run. Init-only noise std measured earlier is 0.099 meV, but data order and dropout seed were fixed in those runs, so true run noise is larger.
- The tail was used to choose among earlier levers, so absolute tail numbers are optimistic; the paired difference is not.
- Attribution, if the gate passes, is limited to "weight decay at this value"; we did not test other values.

## Opus pre-submit review (2026-11-23)
Verdict SUBMIT, no blockers. Notes kept: a result near -1.0 meV is weak evidence (one run, scoring-noise interval only), so only the -1.5 bar is read as adoption-worthy; do not re-sync the remote runner from HEAD (the remote runs the registered version, sha `120d5571...5b37`); matching decay per pass assumes the 500K recipe had the same warmup-plus-cosine schedule (not checked).
