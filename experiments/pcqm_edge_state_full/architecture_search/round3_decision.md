# Round 3 Radical Context Decision

On 2026-08-28, both seed-42 radical-context candidates completed the fixed
official-train-only 100K/10K development protocol. All expected artifacts,
10,000 unique source indices, finite predictions, byte counts, and SHA256
values passed local acceptance. Metrics were independently recomputed from the
retrieved predictions.

| Candidate | Overall MAE | Delta vs control | Radical delta | Non-radical delta | Decision |
|---|---:|---:|---:|---:|---|
| `radicalctx16` | 0.149388 eV | -0.000674 eV | +0.000422 eV | -0.000765 eV | Reject at gate |
| `radicalctx32` | 0.149403 eV | -0.000660 eV | +0.000097 eV | -0.000722 eV | Reject at gate |

Both candidates improved overall and non-radical MAE while keeping radical
regression below the allowed `0.002 eV`. Neither reached the predeclared
`0.002 eV` overall-improvement threshold, so seeds 43 and 44 were not opened.
No full-data run is authorized from this evidence.

The mechanism remains a useful bounded result: a closed-shell identity path
was materially safer than increasing depth, edge width, or replacing the
complete categorical stem. Its observed gain is nevertheless too small to
separate from seed variance without violating the promotion protocol.

This decision closes the single-model portion of the Kaggle architecture
search. A later read-only equal-ensemble analysis opened a separately frozen
multi-model confirmation question in `round4_protocol.md`. Raw outputs and logs are retained under
`platforms/_records/kaggle/training/pcqm_official_architecture_search_round3_20260828/`;
submission identities and artifact hashes are in `round3_submission.json`.
