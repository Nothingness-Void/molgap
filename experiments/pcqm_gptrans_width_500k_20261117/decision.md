# Decision: GPT-T 500K capacity question (width, FFN)

Status: CLOSED under the frozen rules in [protocol.md](protocol.md). All 8 endpoints were scored once on `tail_holdout_v1` (128,606 rows); official valid, test and the sealed 20K were not read. Evidence: [results/](results/) (verdict.json, per-endpoint tail_*.json, retrieval_manifest.json, log excerpts, start_checks_note.md). Verdict code: `width_verdict.py` (sha256 in verdict.json).

## Result
Neither W320 nor F3 is nominated. The L20 calibration reads NO_SIGNAL, so by the pre-registered table the zero results must not be used to close the route.

## Facts
Tail MAE (meV), ema_model.pt of each run, 60 passes, rows [0, 500,000):

| Arm | s42 | s43 | Mean gain vs R | Gain s42 / s43 | Paired 99% upper bound | Outcome |
|---|---|---|---|---|---|---|
| R | 110.913 | 110.924 | - | - | - | base |
| L20 | 111.064 | 111.652 | -0.440 | -0.151 / -0.728 | +0.830 | NO_SIGNAL (calibration) |
| W320 | 113.011 | 113.601 | -2.387 | -2.097 / -2.676 | +2.788 | NOT_NOMINATED |
| F3 | 110.788 | 110.324 | +0.363 | +0.125 / +0.600 | +0.022 | NOT_NOMINATED |

- Gain is positive when the arm is better. Bar for nomination: gain >= 1.5 meV, both seeds better, bound below 0.
- F3 fails the gain bar (0.36 vs 1.5); its bound (+0.022 meV) is also just above 0. W320 is worse than R in both seeds.
- L20 gained -0.44 meV (needs >= 0.9 for PROXY_VALID, <= 0.5 gives NO_SIGNAL). At full scale depth 20 gained +0.89 meV, so this 500K run did not reproduce that direction.
- R s42 vs s43 differ by 0.011 meV. Against the earlier B1024 500K tail MAEs of the same inits (111.672, 110.958): -0.759 and -0.034 meV. The s42 gap is larger than the R seed gap; the earlier recipe identity was not re-verified, so this is a descriptive check only.
- Pre-score check passed for all 8 runs: `train_rows` 500000, `max_shards` 10, `target_mean_eV` 5.413434769952119. All 32 retrieved files match the remote sha256 (see retrieval_manifest.json); local checkpoint sha equals the remote ema_model.pt sha. All 8 tasks were COMPLETE with exit 0.
- Scoring ran with the fixed scorer (branch `codex/infra/tail-eval-guard-20261117`, commit 91bc4155) on an RTX 5060, torch 2.7.1+cu128; the runs trained on A100 with torch 2.7.1+cu126. `run_config.json` holds no `torch_version`, so the scorer's torch alignment did not apply here.

## Inference (not measured)
- At 500K, 60 passes and this recipe, a wider model (W320) did not help and may hurt; a wider FFN (F3) is at best a small gain within noise. This is one budget, two seeds.
- Because L20 shows no gain and full scale does, 500K may not carry the capacity signal. The protocol therefore forbids reading W320 and F3 as closing the width/FFN route. W320's loss is the largest effect in the table (about 2.4 meV, both seeds), which is an observation, not a closure.
- Seed noise here is init-only (sampler and dropout fixed), so the two-seed spread understates run-to-run noise. The row bootstrap covers tail-row noise only.

## Selection bias note
Tail was used here to choose between arms (nominate or not). That use is optimistic for any arm that passes, because the same rows pick and score it. No arm passed, so the bias did not change the outcome. Each endpoint was scored once, no snapshot was chosen and nothing was tuned on tail. Tail MAE is about 7-8 meV higher than the 50K development scale and must not be compared with it.

## Follow-up
No full-scale run follows from this result. Any later width or FFN test needs its own question and decision.

## RML closure
Terminal word: `INCONCLUSIVE`, failure_mode `discriminator_missing`. The L20 calibration reads NO_SIGNAL, so the protocol forbids reading W320 or F3 as closing the width/FFN route; it is also not a positive. It is therefore not NEGATIVE. effect_meV and ci_meV are null: no paired same-job reference file with `pairs[]` exists, and the numbers above are cross-arm tail MAE values that stay in this prose.

Roles. official_validation, test_dev and test_challenge are `untouched` on structural evidence: the scoring files record `protected_roles_read` false for all three. `tail_holdout_v1` is `consumed` with a `metric_computed` event (each of 8 endpoints scored once; the event hash is the `tail_source_idx_sha256` in `results/tail_*.json`). No development role was used (see `role_plan.json`). Costs are not recorded (measurement_missing).

Closure route. `legacy-terminal` plus `terminal-pipeline`, once per action A001..A008 of the trajectory, with this one decision. Evidence is `results/`. Checkpoints and prediction files stay outside the repo; their paths and sha256 are in `results/verdict.json` and `results/retrieval_manifest.json`. Git: the frozen trajectory holds `source_commit` "x" for every action.
