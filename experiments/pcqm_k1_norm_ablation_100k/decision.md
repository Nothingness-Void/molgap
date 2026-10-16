# K1 Norm Ablation Terminal Decision - 2026-10-16

**NEGATIVE_UNDER_CONTRACT; retain BatchNorm baseline, no adoption or scale-up.**

Exact Kaggle2 job `kaseichou/molgap-k1-norm-ablation-100k-s42`, version 1, completed both fresh arms. The raw entry source matches the frozen package `e71d12b00a30765d5003b42255058d6fcc7668f35352306f07a5577e0d717ba3`. Both native qualification certificates and CPU-only artifact inspections pass: 40 epochs, 31,240 optimizer steps and 3,998,720 optimizer-batch sample presentations each. Best live checkpoints are epoch 40 (BN) and epoch 39 (AdaRMSNorm). See [summary](summary.json), [acceptance BN](acceptance_k1_single_bn.json) and [acceptance AdaRMSNorm](acceptance_k1_ada_rmsnorm.json).

| Arm | Backbone Normalization | Best Dev Gap MAE (eV) | Selected Pass / Step | Final Train MAE (eV) | Worker Wall (h) |
|---|---|---:|---:|---:|---:|
| `k1_single_bn` (Control) | 18x BatchNorm1d | **0.141294** (141.294 meV) | Pass 40 / 31240 | 0.098231 (98.23 meV) | 1.4135 |
| `k1_ada_rmsnorm` (Candidate) | 18x AdaRMSNorm | 0.145184 (145.184 meV) | Pass 39 / 30459 | **0.087409** (87.41 meV) | 1.4788 |

Scientific metric evaluation:
- Candidate difference $\Delta\text{MAE} = \text{MAE}(\text{ada}) - \text{MAE}(\text{bn}) = \mathbf{+0.003890\text{ eV}} = \mathbf{+3.890\text{ meV}}$ (Candidate degradation).
- Paired Standard Error: $\mathbf{0.539\text{ meV}}$.
- 10,000-resample paired bootstrap 95% Confidence Interval: $\mathbf{[+2.828, +4.949]\text{ meV}}$. The entire interval is strictly positive and bounded far away from zero.
- Candidate advantage: $\mathbf{-3.890\text{ meV}}$, failing the prospectively frozen non-inferiority gate ($\Delta\text{advantage} \ge -0.5\text{ meV}$) by $3.39\text{ meV}$.
- AdaRMSNorm is also +4.62% slower in process wall time (5323.7s vs 5088.7s).

Measured allocation entry cost is **3.100714 T4 device-hours** (1.550357 wall-hours on 2 observed T4 GPUs) across both arms. Platform queue and pre/post-entry release tails are excluded.

Disposition:
- `k1_ada_rmsnorm`: **NEGATIVE_UNDER_CONTRACT**.
- `k1_single_bn`: **CLOSED** (control baseline retained).
- Retain `BatchNorm1d` as the standard normalization layer for `neural_atom_k1`. No 500K training, retry, or production change is released. Read [attribution](attribution.md) before considering other normalization or regularization variants.
