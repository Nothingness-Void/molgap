# Phase 10 LoRA v3 UQ Decision

Date: 2026-07-06

## Setup

- Base: `phase8_expansion_hybrid`.
- Adapter: v3 Encoder-LoRA, `GPS + SchNet + Fusion`, rank 4.
- Ensemble members: 5 seeds (`42, 1, 2, 3, 4`).
- Calibration split: same OE62 scaffold validation split used by the LoRA run.
- Test split: OE62 scaffold test, 695 molecules.
- UQ: ensemble standard deviation, scaled on the validation split.
- OOD: Euclidean kNN distance in the v3 384-d B3LYP embedding space.

## Accuracy And Calibration

Compared with the calibrated v3 LightGBM Delta baseline:

| target | LightGBM Δ MAE | LoRA ensemble MAE | LoRA R2 | LoRA mean sigma | LoRA ENCE | LoRA 1σ cov | LoRA 2σ cov |
|---|---:|---:|---:|---:|---:|---:|---:|
| HOMO | 0.184 | **0.170** | 0.856 | 0.287 | 0.271 | 0.771 | 0.951 |
| LUMO | 0.214 | **0.177** | 0.914 | 0.318 | 0.280 | 0.814 | 0.953 |
| Gap | 0.291 | **0.237** | 0.909 | 0.360 | 0.259 | 0.757 | 0.917 |

Average MAE is `0.194` eV. Gap improves `0.291 -> 0.237` eV versus the
LightGBM UQ baseline.

## OOD Signal

Embedding-distance OOD still carries signal for the LoRA ensemble:

- Spearman distance vs Gap absolute error: `0.175`.
- Gap MAE near -> far distance decile: `0.183 -> 0.421` eV.
- OOD fraction at fit-set p95 threshold: `3.6%`.

## Decision

LoRA now has a usable first UQ layer and is the best accuracy path. It should be
treated as the high-accuracy candidate for small/medium inference batches.

Do not blindly replace the LightGBM deployment baseline for database-scale runs
yet: LoRA ensemble inference requires 5 full GPS+SchNet+Fusion forward passes,
while LightGBM Delta reuses one B3LYP forward plus cheap tree heads. The next
engineering gate is speed benchmarking and a batch inference wrapper.

## Artifacts

- Metrics: `results/phase10_lora_v3/lora_uq_metrics.json`
- Predictions: `results/phase10_lora_v3/lora_uq_predictions.csv`
- Reliability plots: `results/phase10_lora_v3/reliability_{homo,lumo,gap}.png`
- OOD reference: `results/phase10_lora_v3/ood_reference.npz`
- Evaluation script: `scripts/phase9/eval_encoder_lora_uq.py`
