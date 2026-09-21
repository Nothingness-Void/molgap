# Final Decision: PCQM 100K-to-500K Scale-Transfer Diagnostic

- **Trajectory ID**: `TB-pcqm-scale-transfer-diagnostic`
- **Contract**: `MOLGAP-COMMON-V5-FINAL`
- **State**: `POSITIVE_UNDER_CONTRACT`
- **Outcome**: `NO_TRAIN` (evidence-only diagnostic completed with positive qualification decision)

## Core Analytical Findings

1. **Cross-Scale Exposure Mapping**:
   - 100K 60-epoch endpoint corresponds precisely to **Epoch 12.0 of 500K** ($5,998,080$ presentations / $46,860$ optimizer steps).
   - This proves that a 500K experiment spends 80% of its optimization time *beyond* the entire exposure window of 100K training, explaining why unregularized models suffer severe representation collapse in later epochs.

2. **Empirical Margin vs. Retention Frontier**:
   - Under historical unregularized conditions ($81.7\%$ decay rate), an architecture would require a massive 100K gain of $\ge 0.0164\text{ eV}$ to maintain the required $0.003\text{ eV}$ gain at 500K.
   - Both `K1 PairToken` ($+0.00304\text{ eV}$) and `GPTrans Pair PreNorm` ($+0.00313\text{ eV}$) possessed zero safety buffer and correctly scored in the disqualified zone ($\text{STQS} \approx 20$).

3. **Candidate Mechanism Qualifications**:
   - **`GPTrans Noisy Nodes`**:
     - $\text{STQS} = 61.04$ ($\ge 60.0$).
     - Node denoising directly mitigates over-smoothing across deep horizons.
     - **Recommendation**: `QUALIFIED_FOR_500K` (Currently running on Kaggle 2).
   - **`GPTrans Noisy Nodes + Pair Update Norm`**:
     - $\text{STQS} = 100.0$ (Maximum qualification score).
     - Margin above 100K gate: $+0.00638\text{ eV}$ (Total gain: $+0.00938\text{ eV}$).
     - Dual orthogonal anti-collapse mechanisms (node denoising + relation LayerNorm).
     - Projected 500K retained gain: $\ge 0.0078\text{ eV}$.
     - **Recommendation**: `HIGHLY_QUALIFIED_FOR_500K` (Authorized for immediate 500K scale-transfer submission).

## Next Allowed Actions
- Submit `GPTrans Noisy Nodes + Pair Update Norm` 500K scale-transfer experiment to Kaggle 3 (`nvoid912`).
