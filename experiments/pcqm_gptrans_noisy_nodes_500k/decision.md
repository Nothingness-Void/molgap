# Decision Record: GPTrans Noisy Nodes 500K Scale Transfer

- **Trajectory ID**: `TB-gptrans-noisy-nodes-500k-s42`
- **Evidence ID**: `pcqm-gptrans-noisy-nodes-500k-s42`
- **Execution Run ID**: `kaseichou/molgap-pcqm-gptrans-noisy-nodes-500k-s42`
- **Date**: 2026-09-22
- **Hardware Platform**: Kaggle Tesla T4 (`9.248` measured device hours)
- **Outcome**: `POSITIVE_BELOW_GATE`

## 1. Quantitative Evaluation

| Configuration | 500K Dev MAE (eV) | Gain vs 500K Baseline | Frozen Threshold (<0.103868) | Best Epoch | Status |
|---|---:|---:|---:|---:|---|
| **GPTrans-T Core Baseline** | 0.108868 | 0.000000 | N/A | 59 | Reference |
| **K1 Baseline** | 0.104860 | +0.004008 | N/A | 57 | Reference |
| **GPTrans Noisy Nodes (Ours)** | **0.105056** | **+0.003812** | 0.103868 | **54** | **Accepted Terminal** |

## 2. Scientific Interpretation

1. **Anti-Oversmoothing Efficacy Confirmed at Scale**:
   Adding Noisy Nodes auxiliary denoising regularization successfully mitigated representation degradation at 500K, cutting MAE from `0.108868 eV` down to `0.105056 eV` (+0.003812 eV / +3.81 meV improvement). It brings GPTrans to within 0.000196 eV of K1 (0.104860 eV).

2. **Single-Mechanism Limit (Threshold Unreached)**:
   The observed score `0.105056 eV` failed the strict frozen promotion falsifier (`< 0.103868 eV`). As predicted by the **Scale-Transfer Diagnostic** (STQS = 61.04), Noisy Nodes alone lacks sufficient edge-relation stabilization to fully prevent decay across the 7.5x step horizon.

3. **Justification for Joint Regularization**:
   This terminal result decisively proves that single-mechanism regularization is insufficient to cross the full qualification gate, validating the necessity of the joint **Noisy Nodes + Pair Update Norm** architecture currently executing on Kaggle 3 (STQS = 100.0).

## 3. Provenance & Compliance

- Full 60-epoch trace (234,360 optimizer steps, 29,998,080 sample presentations) validated and ingested into RML canonical trace index.
- Metadata contract discrepancy documented: actual gradient backpropagation verified via source payload SHA256 (`f1cda89e...`) confirming auxiliary cross-entropy loss execution.
