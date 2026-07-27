# 3D encoder A/B (10k subset, scaffold split, fixed budget)

| encoder | params | charges | s/epoch | total s | peak mem (MB) | epochs | **solo Gap MAE** | solo Gap R² | solo avg MAE | fusion Gap MAE | fusion Gap R² |
|---|---|---|---|---|---|---|---|---|---|---|---|
| schnet | 1,041,028 | True | 5.30 | 382 | 1497 | 72 | **0.2388** | 0.8892 | 0.1965 | 0.2230 | 0.9083 |
| visnet | 1,098,115 | False | 16.40 | 1870 | 3512 | 114 | **0.2339** | 0.8945 | 0.1953 | 0.2191 | 0.9080 |
| tensornet | 786,563 | False | 19.78 | 1997 | 2066 | 101 | **0.2222** | 0.9059 | 0.1829 | 0.2175 | 0.9101 |

- **solo** = 3D encoder trained end-to-end with its own head (leak-free, scaffold split) — primary discriminator.
- **fusion** = shared 2D GPS + this 3D encoder via gated FusionHead.
- charges: SchNet uses Gasteiger charges (deployed form); ViSNet/TensorNet use Z+geometry (native).