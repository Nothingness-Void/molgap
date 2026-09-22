# Protocol: PCQM GPTrans DSAR + Noisy Nodes + Pair Update Norm 500K Scale-Transfer

## 1. Scientific Hypothesis
Diagnostic analysis on the 500K scale identified the readout token as the primary representational bottleneck: GPTrans compresses 20-30 real atom representations (7,680 dimensions) through a single 256-dimensional virtual token, discarding fine-grained atomic electron densities. Dual-Stream Attentive Readout (DSAR) with 544-dim dual-stream attentive pooling preserves atomic information and achieved an all-time record MAE of 0.146138 eV (+10.49 meV over baseline) at 100K.

When combined with auxiliary Noisy Nodes atomic denoising ($\sigma=0.15, \alpha=0.10$) and LayerNorm-normalized relation updates, this tri-component architecture (DSAR + Noisy Nodes + Pair Norm) is hypothesized to overcome the capacity bottleneck at 500K and achieve material advantage ($>0.003\text{ eV}$) over the 0.106868 eV baseline.

## 2. Comparators & Falsifier Threshold
- **Reference Comparator**: GPTrans-T 500K baseline (`0.106868 eV`, `TB-matched-500k-v4-three-arm`).
- **Single Mechanism Comparators**:
  - K1 500K baseline (`0.104860 eV`).
  - GPTrans Noisy Nodes + Pair Norm 500K (`0.104812 eV`, `TB-gptrans-noisy-pair-norm-500k-s42`).
- **Required Material Gain**: $\ge 0.003\text{ eV}$ over reference comparator.
- **Falsifier Threshold**: $\text{MAE} < 0.103868\text{ eV}$.
- If the 60-epoch development MAE fails to beat $0.103868\text{ eV}$, the scale-transfer hypothesis is falsified.

## 3. Training Contract
- **Dataset**: `nvoid912/pcqm4mv2-ogb-fixed-500k-scnet-v1` (500,000 train rows, 50,000 internal development rows).
- **Architecture**: `GPTransDSARNoisyNodesPairNorm` ($5,375,961$ parameters).
- **Optimizer**: AdamW, Cosine decay schedule over 60 epochs ($4\times 10^{-4} \rightarrow 1\times 10^{-6}$).
- **Batch Size**: 128, drop last.
- **Precision**: FP32 deterministic algorithms.
- **Platform**: Kaggle 3 (`nvoid912`, Nvidia Tesla T4).
