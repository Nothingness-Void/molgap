# Decision Record: GPTrans Noisy Nodes + Pair Update Norm 500K Scale Transfer

- **Trajectory ID**: `TB-gptrans-noisy-pair-norm-500k-s42`
- **Evidence ID**: `pcqm-gptrans-noisy-pair-norm-500k-s42`
- **Execution Run ID**: `nvoid912/molgap-pcqm-gptrans-noisy-pair-norm-500k-s42`
- **Date**: 2026-09-22
- **Hardware Platform**: Kaggle Tesla T4 (`10.230` measured device hours)
- **Outcome**: `NEGATIVE_UNDER_CONTRACT` (numerically positive, below the frozen gate)

## 1. Quantitative Evaluation

| Configuration | 500K Dev MAE (eV) | Gain vs 500K Baseline | Frozen Threshold (<0.103868) | Best Epoch | Status |
|---|---:|---:|---:|---:|---|
| **GPTrans-T Core Baseline** | 0.106868 | 0.000000 | N/A | 59 | Reference |
| **K1 Baseline** | 0.104860 | +0.002008 | N/A | 57 | Reference |
| **GPTrans Noisy Nodes (Single)** | 0.105056 | +0.001812 | 0.103868 | 54 | Negative Under Contract |
| **GPTrans Noisy Nodes + Pair Norm (Joint)** | **0.104812** | **+0.002056** | 0.103868 | **56** | **Accepted Terminal** |

## 2. Scientific Interpretation

1. **Numerically Matching K1 at Scale**:
   The joint orthogonal regularization (auxiliary Noisy Nodes atom denoising at $\sigma=0.15, \alpha=0.10$ combined with Pair Update LayerNorm) reached `0.104812 eV`, beating the K1 baseline (`0.104860 eV`) by only `0.048 meV` and improving by `2.056 meV` over the frozen GPTrans-T reference.

2. **Threshold Falsifier Determination**:
   The score `0.104812 eV` did not clear the pre-registered `3.0 meV` promotion threshold (`< 0.103868 eV`). Under the strict fail-closed contract, this outcome is `NEGATIVE_UNDER_CONTRACT`; its numerical improvement remains useful attribution evidence only.

3. **Implications for Architecture Strategy**:
   Pair Update Norm adds only `0.244 meV` over Noisy Nodes at 500K. The joint regularizers therefore do not provide a material scale-transfer gain. Readout information loss remains a separate prospective question for the already registered DSAR experiment, not a conclusion established by this result.

## Reference-metric erratum

The original post-run decision and V5 envelope accidentally transcribed the
matched GPTrans-T comparator as `0.108868 eV`. The frozen training contract,
prospective trajectory, and matched-500K authority all specify `0.106868 eV`.
This correction changes only derived gains and the contract disposition; no
training output, prediction, artifact hash, or threshold changed.

## 3. Provenance & Compliance

- Full 60-epoch trace (234,360 optimizer steps, 29,998,080 sample presentations) validated and ingested into RML canonical trace index.
- All artifact hashes verified locally against downloaded Kaggle execution outputs.
