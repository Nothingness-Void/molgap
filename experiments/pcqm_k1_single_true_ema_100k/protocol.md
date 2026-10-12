# Protocol: K1 Single-Forward True-EMA 100K Screen

## 1. Scientific Question
Does a single-forward K1 architecture with step-wise True EMA (`decay=0.9999`) and post-selection Clean-BN calibration achieve non-inferiority or superiority to the established 100K `mean2` reference (0.140257 eV) on PCQM4Mv2 while reducing training step computation by >=35%?

## 2. Hypothesis
- **Observed Deficiency**: Historical K1 training relies on two forward passes per optimization step (either consistency loss or 2-pass mean), adding 40%~70% computational overhead per step. Furthermore, training-time Dropout perturbs BatchNorm running statistics, deteriorating validation MAE by ~1.25 meV. Finally, lack of step-wise True EMA deprived K1 of the ~1.94 meV generalization benefit observed in GPTrans-T.
- **Changed Mechanism**:
  1. Single-forward supervised loss ($L_1$ loss on normalized Gap), eliminating the second forward pass entirely.
  2. True step-wise EMA (`BoundForeachEMA`, decay 0.9999) tracking parameter trajectories to smooth high-frequency gradient noise.
  3. Clean-BN recalibration on 16,384 training molecules without Dropout to restore uncorrupted running mean and variance for evaluation.
- **Cheapest Falsifier**: A complete 40-epoch 100K PCQM4Mv2 training run evaluated on the fixed 50K development set under strictly frozen seed 42, physical batch size 128, and Cosine Annealing learning rate schedule.

## 3. Strict Reference Comparator
- Reference model: Fresh `mean2` control from `pcqm_k1_clean_second_100k` (seed 42, 40 epochs, batch size 128).
- Reference Development MAE: **0.140257 eV**.

## 4. Pre-registered Acceptance & Decision Gates
1. **Cost Gate**:
   - Single-forward training time per step must be $\le 65\%$ of the historical `mean2` per-step time (measured natively on identical hardware).
2. **Quality Gate**:
   - **Superiority Gate**: $\Delta\text{MAE} = \text{MAE}_{\text{ref}} - \text{MAE}_{\text{cand}} \ge 0.003\text{ eV}$ (3.0 meV) advances candidate directly to 500K scale-up candidate.
   - **Non-Inferiority Gate**: $\Delta\text{MAE} \ge -0.0005\text{ eV}$ (-0.5 meV) with verified $\ge 35\%$ speedup qualifies candidate as the standard low-cost K1 baseline.
   - **Negative Gate**: If Clean-BN EMA fails non-inferiority ($\Delta\text{MAE} < -0.0005\text{ eV}$), route is closed as `NEGATIVE_UNDER_CONTRACT`.

## 5. Execution Contract
- Dataset: `pcqm4mv2-ogb-fixed-100k-v1` (100K train, 50K development).
- Architecture: `neural_atom_k1` (3,658,817 parameters).
- Seed: 42 (initialization SHA256: `8ef6d4ba1abdca04d8740a11ec0e04587358117b1e9f9534b4fc19d2b6caedbd`).
- Optimizer: AdamW (lr=4e-4, weight_decay=1e-5, clip_grad_norm=1.0).
- Schedule: CosineAnnealingLR (T_max=31,240, eta_min=1e-6).
- Batch Size: 128 (fixed physical batch, 781 steps/epoch).
- Epochs: 40.
