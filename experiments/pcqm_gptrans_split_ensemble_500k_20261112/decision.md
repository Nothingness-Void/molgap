# GPTrans-T 500K 30+30 against 60: decision

Rule frozen in [protocol.md](protocol.md); applied unchanged. Both arms retrieved with `--all-arms --strict`.
Scored once each on tail_holdout_v1 (128,606 rows), selected_model.pt, no protected role read.
Dev MAE check passed (s43 102.0275, s44 101.8654 meV).

| Quantity | meV |
|---|---:|
| split_s43 tail MAE | 110.000 |
| split_s44 tail MAE | 109.800 |
| equal-weight mean of the two | 106.574 |
| 60-pass comparator (mean of four singles) | 107.94 |
| Delta = fused - 107.94 | -1.366 |

Verdict by protocol: -3 < Delta < +3, so equivalent. Delta is outside 1.5 <= |Delta| <= 4.5, so no second pair
(45/46) is licensed. Member residual correlation (descriptive): Pearson 0.906, Spearman 0.846.
Records: D:/molgap-artifacts/tail_holdout_eval_gptrans_split_500k_s43.json, _s44.json,
tail_fusion_gptrans_split_500k.json, tail_residual_gptrans_split_500k.json.
