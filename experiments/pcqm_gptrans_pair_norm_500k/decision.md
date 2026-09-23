# Decision Record: GPTrans Pair Update Norm 500K Scale Transfer

- **Trajectory ID**: `TB-gptrans-pair-norm-500k-s42`
- **Evidence ID**: `pcqm-gptrans-pair-norm-500k-s42`
- **Execution Run ID**: `nvoid912/molgap-gptrans-pair-update-norm-500k-v1`
- **Date**: 2026-09-23
- **Hardware Platform**: Kaggle Tesla T4 (`9.318` measured device hours)
- **Outcome**: `NEGATIVE_UNDER_CONTRACT` (numerically positive, below the frozen gate)

## 1. Quantitative Evaluation

| Configuration | 500K Dev MAE (eV) | Gain vs 500K Baseline | Frozen Threshold (<0.103868) | Best Epoch | Status |
|---|---:|---:|---:|---:|---|
| **GPTrans-T Core Baseline** | 0.106868 | 0.000000 | N/A | 59 | Reference |
| **K1 Baseline** | 0.104860 | +0.002008 | N/A | 57 | Reference |
| **GPTrans Pair Update Norm (Single)** | **0.105728** | **+0.001140** | 0.103868 | **56** | **Accepted Terminal** |
| **GPTrans Noisy Nodes (Single)** | 0.105056 | +0.001812 | 0.103868 | 54 | Reference Comparison |
| **GPTrans Noisy Nodes + Pair Norm (Joint)** | 0.104812 | +0.002056 | 0.103868 | 56 | Reference Comparison |

## 2. Scientific Interpretation

1. **Independent Normalization Gain at Scale**:
   Parameter-free normalization of newly computed pair updates via LayerNorm achieves `0.105728 eV` at Epoch 56, providing an isolated single-variable gain of `+1.140 meV` (`+0.001140 eV`) over the frozen GPTrans-T baseline (`0.106868 eV`).

2. **Threshold Falsifier Determination**:
   The score `0.105728 eV` did not clear the pre-registered `3.0 meV` material promotion threshold (`< 0.103868 eV`). Under the strict fail-closed contract, this outcome is `NEGATIVE_UNDER_CONTRACT`; its numerical improvement remains useful single-variable attribution evidence.

3. **Disentangling Joint Regularization**:
   Comparing the isolated Pair Norm gain (`+1.140 meV`) with isolated Noisy Nodes (`+1.812 meV`) and their joint combination (`+2.056 meV`) reveals sub-additive diminishing returns ($1.812 + 1.140 = 2.952 > 2.056\text{ meV}$), confirming that both regularizers partially address representation collapse/co-adaptation during extended 500K training.

## 3. Provenance & Compliance

- Full 60-epoch trace (234,360 optimizer steps, 29,998,080 sample presentations) validated and ingested into RML canonical trace index.
- All artifact hashes verified locally against downloaded Kaggle execution outputs.
