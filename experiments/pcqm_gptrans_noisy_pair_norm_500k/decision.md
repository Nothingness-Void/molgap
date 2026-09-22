# Decision Record: GPTrans Noisy Nodes + Pair Update Norm 500K Scale Transfer

- **Trajectory ID**: `TB-gptrans-noisy-pair-norm-500k-s42`
- **Evidence ID**: `pcqm-gptrans-noisy-pair-norm-500k-s42`
- **Execution Run ID**: `nvoid912/molgap-pcqm-gptrans-noisy-pair-norm-500k-s42`
- **Date**: 2026-09-22
- **Hardware Platform**: Kaggle Tesla T4 (`10.230` measured device hours)
- **Outcome**: `POSITIVE_BELOW_GATE`

## 1. Quantitative Evaluation

| Configuration | 500K Dev MAE (eV) | Gain vs 500K Baseline | Frozen Threshold (<0.103868) | Best Epoch | Status |
|---|---:|---:|---:|---:|---|
| **GPTrans-T Core Baseline** | 0.108868 | 0.000000 | N/A | 59 | Reference |
| **K1 Baseline** | 0.104860 | +0.004008 | N/A | 57 | Reference |
| **GPTrans Noisy Nodes (Single)** | 0.105056 | +0.003812 | 0.103868 | 54 | Positive Below Gate |
| **GPTrans Noisy Nodes + Pair Norm (Joint)** | **0.104812** | **+0.004056** | 0.103868 | **56** | **Accepted Terminal** |

## 2. Scientific Interpretation

1. **Surpassing K1 at Scale**:
   The joint orthogonal regularization (auxiliary Noisy Nodes atom denoising at $\sigma=0.15, \alpha=0.10$ combined with Pair Update LayerNorm) reached `0.104812 eV`, beating the K1 baseline (`0.104860 eV`) and improving by `+4.06 meV` over unregularized GPTrans Core.

2. **Threshold Falsifier Determination**:
   Despite setting a new 500K record for the GPTrans family, `0.104812 eV` did not clear the pre-registered 5.0 meV promotion threshold (`< 0.103868 eV`). Under the strict fail-closed contract, this outcome is recorded as `POSITIVE_BELOW_GATE`.

3. **Implications for Architecture Strategy**:
   While joint node and relation regularization successfully suppresses representation collapse and beats single mechanisms, readout information bottleneck (the single virtual token discarding 7,680 dimensions of atomic representations) remains an active limitation at scale. The Dual-Stream Attentive Readout (DSAR) architecture tested at 100K directly addresses this bottleneck.

## 3. Provenance & Compliance

- Full 60-epoch trace (234,360 optimizer steps, 29,998,080 sample presentations) validated and ingested into RML canonical trace index.
- All artifact hashes verified locally against downloaded Kaggle execution outputs.
