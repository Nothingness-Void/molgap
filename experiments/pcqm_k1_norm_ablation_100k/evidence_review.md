# Evidence Review - K1 Normalization Ablation 100K

## Historical K1 Baselines (PCQM4Mv2 100K Pure 2D)
1. **Single-Forward K1**:
   - Live endpoint: 0.13937 eV
   - Raw EMA endpoint: 0.15653 eV (severe degradation)
   - Clean-BN calibrated EMA endpoint: 0.14310 eV
   - Gap: The live point is favorable, but standard EMA suffers from BatchNorm running statistics mismatch (a 3.73 meV penalty). The clean-BN calibration is non-causal, requiring post-hoc batch presentation passes that complicate reproducible training and inference.
2. **Mean2 K1**:
   - Live endpoint: 0.14026 eV (two-forward stochastic L1 loss).
   - Requires 2x forward compute per optimizer step.

## Theoretical and Literature Justification
- **PPGT (Plain Transformers Can be Powerful Graph Learners)** (Ma et al., TMLR 2026, arXiv:2504.12588):
  - Table 4 demonstrates 0.0856 eV PCQM4Mv2 pure-2D validation MAE with a simplified plain Transformer.
  - Section 3 & 5.4 ablation highlights AdaRMSNorm (Adaptive Root-Mean-Square Normalization):
    $$\text{AdaRMSN}(x) = \frac{x}{\text{RMS}(x)} \cdot \gamma'(x), \quad \gamma'(x) = \frac{1}{\sqrt{D}}\|\alpha \odot x + \beta\|_2$$
  - With $\alpha=0, \beta=1$ at initialization, AdaRMSNorm recovers identity-scaled RMSNorm without batch-level dependencies.
  - Unlike BatchNorm, AdaRMSNorm possesses zero running statistics and zero buffers.
  - Token-level / node-level normalization is completely deterministic and identical across training, evaluation, and EMA tracking.
  - Parameter parity: Exactly 384 parameters per layer ($192 \times 2$), matching standard BatchNorm1d(192) ($192 \times 2$) with 0 buffers instead of 3.

## Research Question
Does replacing K1's 18 BatchNorm layers in `LocalGPSBlock` with AdaRMSNorm eliminate EMA buffer drift without requiring Clean-BN calibration, while matching or improving upon the single-forward Live baseline ($\Delta\text{MAE} \ge -0.5\text{ meV}$ non-inferiority)?
