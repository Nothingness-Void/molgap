# K1 Normalization Ablation 100K Dual-Arm Comparison

Evaluation of whether replacing K1's 18 BatchNorm layers with AdaRMSNorm (from PPGT, 2025/2026) eliminates EMA buffer degradation, simplifies the state machine, and matches/exceeds the single-forward Live baseline.

- Run ID: `pcqm-k1-norm-ablation-100k-s42`
- Platform: Kaggle1 (Nvidia Tesla T4x2)
- Arms: `k1_single_bn` vs. `k1_ada_rmsnorm`
- Protocol: [protocol.md](protocol.md)
- Authority: [authority.json](authority.json)
- Remote handoff: [REMOTE_HANDOFF.md](REMOTE_HANDOFF.md)
