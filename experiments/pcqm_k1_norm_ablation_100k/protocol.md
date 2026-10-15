# Protocol - K1 Normalization Ablation 100K Dual-Arm Comparison

## User Authorization and Scope
- User explicit directive: Submit a 100K dual-arm experiment on Kaggle1.
- Target hardware: Kaggle1 private Notebook with 2x Nvidia Tesla T4 GPUs (`NvidiaTeslaT4`).
- Allocation envelope: 28,800 seconds (8.0 wall hours) maximum allocation, with 180s cleanup reserve.
- Strict pure 2D: PCQM4Mv2 molecular graph topology (OGB atom9/bond3/RWSE16). No 3D conformer construction or geometric coordinates.

## Dual-Arm Design
1. **Arm 0 (`k1_single_bn`) [Control / Baseline]**:
   - Model: `neural_atom_k1` (version 3) with standard BatchNorm layers in all 9 LocalGPSBlocks (18 BN layers total, 54 buffers).
   - Mode: `reference` (single forward pass per optimizer step).
   - Device: `cuda:0`.
   - Initial state: Seed 42 frozen K1 state (`8ef6d4ba1abdca04d8740a11ec0e04587358117b1e9f9534b4fc19d2b6caedbd`).
2. **Arm 1 (`k1_ada_rmsnorm`) [Candidate]**:
   - Model: `neural_atom_k1` (version 3) with AdaRMSNorm replacing all 18 BatchNorm layers in LocalGPSBlocks (`k1_ada_rmsnorm`).
   - Mode: `ada_rmsnorm` (single forward pass per optimizer step).
   - Device: `cuda:1`.
   - Initial state: Seed 42 frozen K1 state with identical non-BN weights and $\alpha=0, \beta=1$ AdaRMSNorm parameters (`cb6dca79b58dd27d76569b8bd2a9ad5e18008d887352efb61249e925be33be33`).

## Common Training Hyperparameters
- Shard / manifest: `pcqm_fixed_100k_v1/manifest.json` (SHA256: `1b0e8fd579ab1cb86c02e833e7ad284b4af7582b059f912a77853fdccf3ede6d`).
- Train rows: 100,000 (indices 0..99,999).
- Development rows: 50,000 (indices 100,000..149,999).
- Epochs: 40 passes.
- Steps per pass: 781 optimizer steps (physical batch size 128, drop-last).
- Total optimizer steps: 31,240 updates.
- Total sample presentations: 3,998,720 presentations.
- Optimizer: AdamW (lr=0.0004, weight_decay=1e-5, betas=(0.9, 0.999), eps=1e-8).
- Schedule: CosineAnnealingLR (total_units=40, eta_min=1e-6, epoch-based, 0 warmup).
- Loss: L1 loss in normalized Gap space.
- Target normalization: mean = 5.3383002281188965 eV, std = 1.275090217590332 eV.
- Selection: `best-development-live`.
- EMA decay: 0.999 tracked.

## Evaluation & Decision Gates
- Evaluation metric: PCQM4Mv2 development Gap MAE in eV.
- Non-inferiority tolerance: $\Delta\text{MAE} = \text{MAE}(\text{k1\_single\_bn}) - \text{MAE}(\text{k1\_ada\_rmsnorm}) \ge -0.5\text{ meV}$.
- EMA health check: Verify that EMA development MAE matches or improves live MAE without requiring Clean-BN calibration passes.
- Outcome:
  - If $\Delta\text{MAE} \ge -0.5\text{ meV}$ and EMA is healthy: Adopt AdaRMSNorm as standard normalization layer for K1 and future PPGT architectures.
  - If $\Delta\text{MAE} < -0.5\text{ meV}$: Record terminal result as negative under contract; retain BatchNorm baseline.
