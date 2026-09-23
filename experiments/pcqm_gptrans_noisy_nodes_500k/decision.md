# Decision Record: GPTrans Noisy Nodes 500K Scale Transfer

- **Trajectory ID**: `TB-gptrans-noisy-nodes-500k-s42`
- **Evidence ID**: `pcqm-gptrans-noisy-nodes-500k-s42`
- **Execution Run ID**: `kaseichou/molgap-pcqm-gptrans-noisy-nodes-500k-s42`
- **Date**: 2026-09-22
- **Hardware Platform**: Kaggle Tesla T4 (`9.248` measured device hours)
- **Outcome**: `NEGATIVE_UNDER_CONTRACT` (numerically positive, below the frozen gate)

## 1. Quantitative Evaluation

| Configuration | 500K Dev MAE (eV) | Gain vs 500K Baseline | Frozen Threshold (<0.103868) | Best Epoch | Status |
|---|---:|---:|---:|---:|---|
| **GPTrans-T Core Baseline** | 0.106868 | 0.000000 | N/A | 59 | Reference |
| **K1 Baseline** | 0.104860 | +0.002008 | N/A | 57 | Reference |
| **GPTrans Noisy Nodes (Ours)** | **0.105056** | **+0.001812** | 0.103868 | **54** | **Accepted Terminal** |

## 2. Scientific Interpretation

1. **A Small Numerical Gain Survived at Scale**:
   Adding Noisy Nodes auxiliary denoising regularization reduced MAE from the frozen matched GPTrans-T reference of `0.106868 eV` to `0.105056 eV` (`+0.001812 eV`). It brings GPTrans to within `0.000196 eV` of K1 (`0.104860 eV`), but the retained gain is below the materiality gate.

2. **Single-Mechanism Limit (Threshold Unreached)**:
   The observed score `0.105056 eV` failed the strict frozen promotion falsifier (`< 0.103868 eV`). The outcome is therefore `NEGATIVE_UNDER_CONTRACT`; the positive numerical delta does not qualify the mechanism for promotion.

3. **Implication for Further Work**:
   This terminal result shows that Noisy Nodes alone is insufficient to cross the qualification gate. It does not by itself validate another mechanism or the uncalibrated STQS projection.

## Reference-metric erratum

The original post-run decision and V5 envelope accidentally transcribed the
matched GPTrans-T comparator as `0.108868 eV`. The frozen training contract,
prospective trajectory, and matched-500K authority all specify `0.106868 eV`.
This correction changes only the derived gain and contract disposition; no
training output, prediction, artifact hash, or threshold changed.

## 3. Provenance & Compliance

- Full 60-epoch trace (234,360 optimizer steps, 29,998,080 sample presentations) validated and ingested into RML canonical trace index.
- Metadata contract discrepancy documented: actual gradient backpropagation verified via source payload SHA256 (`f1cda89e...`) confirming auxiliary cross-entropy loss execution.
