# Full Replacement300k Retrain Summary

Date: 2026-06-26

Dataset: `data/raw/phase8_replacement_300k.csv`

## Graph cache

| kind | processed | graphs | failed | elapsed min | output |
|---|---:|---:|---:|---:|---|
| 2d | 300,000 | 300,000 | 0 | 4.0 | `D:/文档/molgap/results/phase8/pyg_2d_graphs_bond_replacement_300k.pt` |
| 3d | 300,000 | 298,957 | 1,043 | 59.0 | `D:/文档/molgap/results/phase8/pyg_3d_graphs_etkdg_replacement_300k.pt` |

## Internal replacement300k test

| model | init | best val MAE | best epoch | test HOMO | test LUMO | test Gap | test avg |
|---|---|---:|---:|---:|---:|---:|---:|
| GPS 2D | `models\gps_2d_300k.pt` | 0.10880 | 0 | 0.09910 | 0.09646 | 0.13053 | 0.10870 |
| SchNet 3D | `models\gnn_schnet_3d_300k.pt` | 0.12230 | 3 | 0.11024 | 0.11159 | 0.14842 | 0.12342 |
| Hybrid FusionHead | GPS+SchNet embeddings | 0.09661 | 49 | 0.08922 | 0.08809 | 0.11503 | 0.09745 |

Internal test note: this split is the replacement300k distribution, so it is not directly comparable to the Phase 7 internal test. Model selection uses the common eval below.

## Common eval versus Phase 7 full baseline

| eval set | P7 avg MAE | replacement avg MAE | delta avg | P7 Gap MAE | replacement Gap MAE | delta Gap |
|---|---:|---:|---:|---:|---:|---:|
| all | 0.14529 | 0.12839 | -0.01690 | 0.17930 | 0.15610 | -0.02320 |
| Phase 7 OOD-1000 | 0.12431 | 0.12144 | -0.00287 | 0.14881 | 0.14479 | -0.00402 |
| P8 targeted hard | 0.16671 | 0.13548 | -0.03123 | 0.21044 | 0.16765 | -0.04279 |

Conclusion: full replacement300k standard FusionHead became the selected v2 base after the P8.7 audit. It improves the shared common eval overall, improves OOD-1000 slightly, and strongly improves the P8 targeted hard slice.

## Artifacts

- `models/phase8_gps_replacement_300k.pt`
- `models/phase8_schnet_replacement_300k.pt`
- `models/phase8_hybrid_fusion_replacement_300k.pt`
- `results/phase8/gps_replacement_300k_embeddings.pt`
- `results/phase8/schnet_replacement_300k_embeddings.pt`
- `results/phase8/fusion_replacement_300k_metrics.json`
- `results/phase8/full_replacement_common_eval_metrics.json`
- `results/phase8/full_replacement_common_eval_predictions.csv`
- `results/phase8/full_replacement_300k_summary.json`
