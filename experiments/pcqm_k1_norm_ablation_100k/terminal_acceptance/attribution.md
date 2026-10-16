# Normalization Replacement Failure-Mode Attribution - 2026-10-16

The prospectively frozen hypothesis was that replacing 18 `BatchNorm1d` layers in `neural_atom_k1` (9 LocalGPSBlocks) with sample-deterministic `AdaRMSNorm` (adaptive per-node RMSNorm conditioned on graph representation) would remove batch-size and batch-composition dependencies without degrading 100K development Gap MAE (frozen non-inferiority gate $\Delta\text{MAE} \ge -0.5\text{ meV}$).

This fresh paired same-job screen decisively falsified the hypothesis:
1. **Generalization Gap Divergence**:
   - `k1_ada_rmsnorm` fits the training set significantly better than `k1_single_bn`: Pass 40 train MAE is **0.08741 eV** (87.41 meV) vs **0.09823 eV** (98.23 meV) for BN, representing a **-10.82 meV lower training error**.
   - However, its development MAE degrades to **0.14518 eV** (145.18 meV) vs **0.14129 eV** (141.29 meV) for BN (**+3.89 meV worse validation error**).
   - Consequently, the train-to-development generalization gap widens from **43.06 meV** (BN) to **57.77 meV** (AdaRMSNorm), a **+14.71 meV expansion**.
2. **Mechanism Attribution**:
   - `neural_atom_k1` is trained with `dropout=0.0` and `attn_dropout=0.0`. Under this design, the batch-statistic fluctuations of mini-batch `BatchNorm1d` serve as a critical implicit stochastic regularizer across packed molecular graphs.
   - Replacing `BatchNorm1d` with per-node `AdaRMSNorm` removes all batch-stochastic noise, transforming the backbone into an entirely deterministic per-sample mapping. Without stochastic regularization or explicit node/edge dropout, the 9-block Transformer overfits the 100K training set.
3. **Execution Cost**:
   - Per-node adaptive scaling in PyTorch adds multiple elementwise reductions and kernel launches per block, resulting in a +4.62% wall time increase (5323.7s vs 5088.7s).
4. **Boundary and Next Step**:
   - The falsification is conclusive under the contract (paired bootstrap 95% CI `[+2.828, +4.949] meV` degradation).
   - Any future exploration of non-batch normalization in `neural_atom_k1` must explicitly incorporate compensatory stochastic regularization (e.g. stochastic depth, DropPath, or node feature noise) under a new prospective trajectory. No automatic retry or scale-up is released.
