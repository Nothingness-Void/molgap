# Prospective trajectory draft: DeMol-CIN 500K serial chain (not in RML)

Draft text only. AGENTS requires a canonical `trajectory.json` (`record_mode = prospective`) before any training.
Gates and admission: `opus_ruling_20261121.md`. Write it after the Opus review and the user's confirmation, with the repo's trajectory tooling; nothing here
touches `research_memory/`. Parent records: the replication trajectories (J1, J2) and the F0 trajectory on the
500K branch (commit `a7c5f7d74`). J3, J4 and J5 have no trajectory yet.

## Family and owner
- Family: `pcqm-demol-cin-ims-replication-20261120` (same family as J1/J2; J3-J5 extend it to 500K).
- Owner: desktop. Branch: `codex/exp/demol-cin-500k-chain-20261121`. Mode: prospective, non-strict (IMS V4 has no certificate layer).

## One trajectory per arm
| Arm | Question | Reference | Cheapest falsifier |
|---|---|---|---|
| J1 + J2 | Does the 100K DeMol-CIN gain replicate on a second seed (s43) on IMS? | J2 GPT-T pair-update-norm, same job | Delta43 (dev) above -1.5 meV: close the 100K lead |
| J3, J4 | Does the DeMol-CIN gain reach 500K? | Kaggle 500K cohort, tail G = 107.940; J5 on IMS | J3 dev no better than J5 (J4 gate, 0.0 meV), or tail Delta above -1.5 meV |
| J5 | Does IMS A100 torch 2.7.1 shift the 500K GPT-T tail by 1.0 meV or less? | Kaggle init_s44, tail 107.273 | |shift| above 2.0 meV: cross-platform 500K comparisons are invalid |

## Fields shared by all arms
- Deficiency: the GPT-T 500K floor, 107.27-108.36 meV on the tail.
- Evidence: 100K dev -4.805 and tail -6.102 meV (s42, one seed, Kaggle); s43 pending in J1/J2.
- Changed mechanism: cellular complex (atoms, bonds with angle links, rings of 3-8, virtual hub).
- Alternative explanations: selection bias on dev (best of 60 passes), init noise (about 1.6 meV at 100K), a 100K-only regime
  (K1 Muon beat GPT-T by 6 meV at 100K tail and lost at 500K; tri-pair dev -1.67 became -0.66).
- Closed neighbours: tri-pair family, width and depth at 500K, levers, late give-back A, Cutlass/GRIT scoring (open, not ours), K1 (sealed).
- Decision changed if positive: nominate a full-scale DeMol question. No adoption.
- Decision changed if negative: close DeMol single-model 500K and full; no retries, no weight-decay or schedule grids.

## Roles
Train rows below 3,250,000 only. Dev = the view's own 50K rows (100K view: 100,000-149,999; 500K view: 500,000-549,999).
`tail_holdout_v1` is scored offline once per selected checkpoint after retrieval. Official valid and test: sealed, not authorized.
Cutlass/GRIT 500K checkpoints get no tail score here.

## Cost events (estimated, not measured)
- One IMS job: 1 A100, 16 cores, 24 h wall, queue H. Measured rate 73.1 points per A100-h; at most 2 resume jobs, so at most
  3 x 24 h x 73.1 = about 5,263 points (re-check with `showlim -c` on submit day).
- Per-item budgets (seconds, start thresholds): J1 7200, J2 4140, J3 28800, J5 15300, J4 7200. Total 62,640 of 84,480 allowed.
  J3 and J4 each need about 4-9 A100-h (rate unmeasured); J4 likely finishes in one resume job.
- Measured fields stay empty until retrieval: A100 step rate, Linux peak memory, queue wait.

## Actions and outcome
- A001 training: one serial chain job (run id `demol500k-chain-20261121`). Outcome ACTIVE. Next allowed: retrieval, offline scoring, then the frozen rules.
- Reopen conditions: terminal evidence only.

## Closure plan
Per-item terminal attribution beside the decision before any unrelated module. Missing comparator, trace or checkpoint means
`insufficient_evidence`, not a conclusion. Rebuild derived RML once after the accepted milestone.
