# Phase 8 Tail Probe 30K Fusion-Head Decision

Date: 2026-06-30

## Setup

- Base model: `phase8_expansion_hybrid` (v3).
- Tail pool: `data/raw/phase8_tail_probe_30k.csv`.
- Usable tail rows: 20,829 unique rows after excluding expansion500k.
- ETKDG-valid tail rows: 20,192 / 20,829.
- Probe: freeze v3 GPS + SchNet encoders, append tail-pool embeddings to
  expansion500k embeddings, train a standard `FusionHead` only.
- New checkpoint: `models/phase8_hybrid_fusion_tail_probe_30k.pt`.

This is a low-cost decision run. It tests whether the new tail labels improve
the learned fusion surface before paying for a full encoder-level retrain.

## Common Eval

Lower MAE is better. Delta is `tail_probe30k_fusion - expansion500k_full`.

| Eval set | v3 avg MAE | tail avg MAE | delta avg | v3 Gap MAE | tail Gap MAE | delta Gap | v3 Gap R2 | tail Gap R2 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| all | 0.105591 | 0.106382 | +0.000791 | 0.125276 | 0.126610 | +0.001334 | 0.956879 | 0.955828 |
| OOD-1000 | 0.113725 | 0.114717 | +0.000992 | 0.133984 | 0.135450 | +0.001466 | 0.964677 | 0.963981 |
| P8 targeted hard | 0.097283 | 0.097869 | +0.000586 | 0.116381 | 0.117581 | +0.001199 | 0.916345 | 0.913960 |

## PCQM4Mv2 Valid Proxy

This is the same leakage-filtered proxy as the v1/v2/v3 audit, with tail rows
also excluded from the training-union filter. It is not an OGB submission.

| Model | n | Gap MAE | delta vs v3 |
|---|---:|---:|---:|
| v1 / P7 | 2,988 | 0.258824 | +0.005762 |
| v2 / replacement300k | 2,988 | 0.251937 | -0.001124 |
| v3 / expansion500k | 2,988 | 0.253062 | 0.000000 |
| tail probe fusion | 2,988 | 0.252272 | -0.000790 |

## Decision

Do not promote the tail-probe fusion head. The new tail data gives a tiny PCQM
proxy improvement versus v3, but it consistently worsens the primary common
eval slices, including the exact P8 hard slice it was meant to improve.

This does not justify a full encoder-level retrain yet. The more likely
interpretation is that 20k tail rows are too small and distribution-skewed for a
fusion-only top-up; more B3LYP-only data has lower ROI than Phase 9 GW
Delta-learning unless a future tail pool is much larger and balanced.

## Artifacts

- Dataset report: `results/phase8/expansion_tail_probe_521k_report.md`
- Graph report: `results/phase8/graph_build_report_tail_probe_30k.json`
- Fusion metrics: `results/phase8/fusion_tail_probe_30k_metrics.json`
- Common eval metrics: `results/phase8/tail_probe_30k_common_eval_metrics.json`
- Common eval predictions: `results/phase8/tail_probe_30k_common_eval_predictions.csv`
- PCQM proxy metrics: `results/phase8/pcqm4mv2_proxy_p7_v2_v3_tail_metrics.json`
- PCQM proxy predictions: `results/phase8/pcqm4mv2_proxy_p7_v2_v3_tail_predictions.csv`
