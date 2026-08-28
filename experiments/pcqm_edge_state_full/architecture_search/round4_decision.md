# Round 4 Equal-Ensemble Decision

On 2026-08-28, the frozen equal average of matched-seed control,
`radicalctx16`, and `radicalctx32` passed the official-train-only three-seed
confirmation protocol.

| Seed | Control MAE | Equal ensemble MAE | Overall delta | Radical delta | Non-radical delta |
|---:|---:|---:|---:|---:|---:|
| 42 | 0.150062 | 0.143317 | -0.006746 | -0.012870 | -0.006239 |
| 43 | 0.150637 | 0.145418 | -0.005219 | -0.016373 | -0.004296 |
| 44 | 0.151366 | 0.144922 | -0.006445 | -0.015920 | -0.005661 |
| Mean | 0.150688 | 0.144552 | -0.006136 | -0.015054 | -0.005399 |

All three seeds improved overall, both subsets improved in every seed, and the
mean overall gain exceeded the predeclared `0.002 eV` threshold. A subsequent
matched-cost control in `round4_matched_cost_decision.md` showed that three
ordinary control seeds perform better than this mixed architecture at the same
three encoder passes. The ensemble gate passed, but radical context is not
promoted as an architecture gain.

The evidence does not establish official PCQM4Mv2 validation or leaderboard
performance. It uses only the fixed 100K/10K roles drawn from official
training rows, and a full-data three-member run was not authorized. Inference
cost is three complete encoder passes, so this is an accuracy-mode candidate,
not a low-cost replacement.

All six radical-context outputs passed count, identity, finite-value, byte,
and SHA256 acceptance. Raw artifacts and logs are retained under
`platforms/_records/kaggle/training/pcqm_official_architecture_search_round3_20260828/`
and
`platforms/_records/kaggle/training/pcqm_official_architecture_search_round4_20260828/`.
Machine-readable metrics are in `round4_acceptance.json`.
