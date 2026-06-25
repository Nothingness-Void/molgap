# Intermediate-layer Fusion Comparison

Date: 2026-06-25

Question: does concatenating intermediate encoder layers (2/4/final) produce a better replacement30k fusion model than the current single head or MoE probes?

## B3LYP replacement30k internal test

| method | scope | best val MAE | test avg MAE | test Gap MAE | delta avg vs single | delta Gap vs single | params |
|---|---|---:|---:|---:|---:|---:|---:|
| Single FusionHead | replacement30k frozen embeddings | 0.13902 | 0.13838 | 0.16251 | +0.00000 | +0.00000 | 203,907 |
| MoE FusionHead (4 experts) | replacement30k frozen embeddings | 0.13871 | 0.13778 | 0.16211 | -0.00060 | -0.00040 | 409,360 |
| Intermediate-layer fusion (2/4/final) | replacement30k frozen layer embeddings | 0.13875 | 0.13719 | 0.16149 | -0.00118 | -0.00102 | 351,363 |
| End-to-end MoE (4 experts) | replacement30k trainable encoders | 0.14362 | 0.14170 | 0.17301 | +0.00332 | +0.01051 |  |

Internal split conclusion: intermediate-layer fusion is the best frozen-embedding head on replacement30k, beating the single head by -0.00118 avg MAE / -0.00102 Gap MAE and MoE by -0.00058 avg MAE / -0.00062 Gap MAE. The gain is real but still small.

## B3LYP common eval: single vs intermediate-layer fusion

| eval set | single avg MAE | layer avg MAE | delta avg | single Gap MAE | layer Gap MAE | delta Gap |
|---|---:|---:|---:|---:|---:|---:|
| All common eval | 0.21612 | 0.21671 | +0.00059 | 0.26721 | 0.26693 | -0.00028 |
| Phase 7 OOD-1000 | 0.20043 | 0.19880 | -0.00163 | 0.24121 | 0.23721 | -0.00400 |
| P8 targeted hard | 0.23210 | 0.23501 | +0.00292 | 0.29370 | 0.29729 | +0.00359 |

Common-eval conclusion: layer fusion improves the Phase 7 OOD-1000 slice, especially Gap (-0.00400), but worsens the P8 targeted hard slice (+0.00292 avg / +0.00359 Gap). Overall avg is slightly worse (+0.00059), while overall Gap is effectively tied/slightly better (-0.00028). This is mixed, not a clean default upgrade.

## GW Delta / LoRA context

These rows are OE62 GW scaffold-test Delta-learning results and are not directly comparable to the B3LYP surrogate MAEs above.

| method | scope | avg MAE | Gap MAE | trainable params | train time s |
|---|---|---:|---:|---:|---:|
| LightGBM / external Delta reference | OE62 GW scaffold-test | 0.23872 | 0.30287 |  |  |
| LoRA FusionHead r8 | OE62 GW scaffold-test | 0.23581 | 0.30292 | 16,920 | 10.4 |
| Encoder LoRA GPS+SchNet+Fusion r4 | OE62 GW scaffold-test | 0.21718 | 0.27169 | 130,368 | 431.0 |

LoRA conclusion: encoder LoRA is useful for Phase 9 GW Delta after the base model is selected. It does not replace the Phase 8 B3LYP base-model decision.

## Decision

- Default next full run stays: full replacement300k with standard single FusionHead, preferably warm-started from Phase 7 GPS/SchNet.
- Do not run full 300k MoE by default: 30k gains are tie-level and the end-to-end MoE pilot is worse.
- Keep intermediate-layer fusion as a low-cost follow-up after full replacement300k embeddings exist. It is worth one head-only run on full embeddings, but not worth delaying the standard full run.
- If revisiting layer fusion, add regularization/projection or test layer choices; current 30k result looks slightly overfit to internal split versus the P8 hard common slice.

Artifacts:

- trainer: `scripts/phase8/train_layer_fusion.py`
- common evaluator: `scripts/phase8/eval_layer_fusion_common.py`
- internal metrics: `results/phase8/layer_fusion_replacement30k_metrics.json`
- common metrics: `results/phase8/layer_fusion_common_eval_metrics.json`
- machine-readable summary: `results/phase8/intermediate_layer_fusion_comparison.json`
