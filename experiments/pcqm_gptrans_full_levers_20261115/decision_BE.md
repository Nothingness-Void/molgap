# Arms B (batch 256) and E (cross-recipe ensemble): tail verdict (2026-11-16)

Rules: [protocol.md](protocol.md). Code `verdict_levers.py` (cf25ee6c, committed before any C/D/B256 number). Evidence: `D:/molgap-artifacts/tail_eval_full_levers_20261116/verdict_levers_final.json`.

Retrieval accepted: both B256 runs COMPLETE 761,700 steps / 60 passes after one resubmission each (jobs 2016619/2016620; the first jobs 2004612/2004613 paused at 59/60 on walltime). Remote sha256 equals local for weights, run_config, completion record, logs; init shas a65034da... (s42) and c95da412... (s43). Scored once each, `ema_model.pt`.

| Run | tail MAE (meV) |
|---|---|
| B256 s42 | 89.439 |
| B256 s43 | 89.256 |

**B: FAIL.** B256 is worse than B1024 for the same init: +1.498 (99% upper 1.883) and +1.410 (1.753) meV. Mean of the two singles 89.347 vs M 87.938. Disclosure: `verdict_levers.py` reports the B-arm *ensemble* MAE (86.926) as `tail_mev` for the bar; the frozen wording is the mean of the two singles (89.347). Both miss the bar M - 1.5 = 86.438, so the outcome is the same; the numbers above use the frozen wording.

**E: FAIL.** Eight-run equal-weight ensemble (4 x B1024, C, D, 2 x B256) 83.340 meV vs best same-recipe ensemble (all four B1024) 83.992: -0.653 (99% upper -0.545), threshold -1.0. Not adopted.

## Summary of the question
Only A passed. Candidate: all four B1024 inits, equal weights, tail 83.992 meV (-1.357 vs the post-hoc 85.349). C and D each gained about 0.8-0.9 meV single-model, below the 1.5 bar; B256 hurts. Tail was used to choose among levers, so 83.992 is optimistic. Cost (measured, A100 h): C 13.82, D 12.93; B256 in `sessions.jsonl` of each retrieved run.
