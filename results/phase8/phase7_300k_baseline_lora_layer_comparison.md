# Phase 7 300k Baseline / LoRA / Intermediate-layer Fusion

Date: 2026-06-25

Scope: original Phase 7 300k data. B3LYP fusion rows use the same ETKDG-aligned effective set (`N=299,629`) and the same deterministic split seed as the P7 ordinary Fusion baseline.

## B3LYP surrogate: original 300k internal test

| method | best val MAE | HOMO MAE | LUMO MAE | Gap MAE | avg MAE | delta avg vs baseline | delta Gap vs baseline |
|---|---:|---:|---:|---:|---:|---:|---:|
| P7 ordinary FusionHead baseline | 0.06709 | 0.06400 | 0.06171 | 0.07563 | 0.06711 | +0.00000 | +0.00000 |
| Intermediate-layer fusion (2/4/final) | 0.06752 | 0.06437 | 0.06189 | 0.07594 | 0.06740 | +0.00029 | +0.00031 |

Conclusion: on original P7 300k, intermediate-layer fusion is a tie/slight loss, not an upgrade. It is +0.00029 avg MAE and +0.00031 Gap MAE worse than the ordinary FusionHead baseline, while using a larger head input (576+576 vs 192+192).

## GW Delta / LoRA context

These LoRA rows fine-tune the P7 base toward OE62 GW targets. They answer the Phase 9 Delta-learning question, not the Phase 7 B3LYP surrogate internal-test question above.

| method | HOMO MAE | LUMO MAE | Gap MAE | avg MAE | trainable params | train time s |
|---|---:|---:|---:|---:|---:|---:|
| LightGBM external Delta reference | 0.19667 | 0.21663 | 0.30287 | 0.23872 |  |  |
| LoRA FusionHead r8 | 0.18644 | 0.21806 | 0.30292 | 0.23581 | 16,920 | 10.4 |
| Encoder LoRA GPS+SchNet+Fusion r4 | 0.18248 | 0.19736 | 0.27169 | 0.21718 | 130,368 | 431.0 |

Decision: keep P7 ordinary FusionHead as the full original-300k baseline. Use encoder LoRA only for the later GW Delta path after the B3LYP base is selected. Intermediate-layer fusion can be tested head-only on future full replacement300k embeddings, but P7 full-scale evidence does not justify replacing the ordinary FusionHead.

Artifacts:

- baseline_metrics: `results/phase7/fusion_optuna_metrics.json`
- layer_metrics: `results/phase8/layer_fusion_phase7_300k_metrics.json`
- layer_model: `results/phase8/layer_fusion_phase7_300k.pt`
- lora_metrics: `results/phase9/lora_fusion_delta_metrics.json`
- encoder_lora_metrics: `results/phase9/encoder_lora_delta_gps_schnet_fusion_r4_metrics.json`
