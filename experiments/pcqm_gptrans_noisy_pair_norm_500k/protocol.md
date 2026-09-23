# Protocol: PCQM GPTrans-T Noisy Nodes + Pair Update Norm 500K Scale-Transfer

## 1. Scientific Hypothesis
Orthogonal joint regularization (auxiliary Noisy Nodes atom denoising at $\sigma=0.15, \alpha=0.10$ combined with LayerNorm-normalized relation updates) yielded a project-record MAE of $0.147245\text{ eV}$ ($+0.009382\text{ eV}$ over baseline) at 100K. With a Scale-Transfer Qualification Score of $100.0$ ($\text{STQS} \ge 60.0$), this joint regularization architecture is hypothesized to resist over-smoothing across the $7.5\times$ optimizer step horizon and maintain material advantage ($>0.003\text{ eV}$) at 500K.

## 2. Comparators & Falsifier Threshold
- **Reference Comparator**: GPTrans-T 500K baseline (`0.106868 eV`).
- **Single Mechanism Comparator**: K1 500K baseline (`0.104860 eV`).
- **Required Material Gain**: $\ge 0.003\text{ eV}$ over reference comparator.
- **Falsifier Threshold**: $\text{MAE} < 0.103868\text{ eV}$.
- If the 60-epoch development MAE fails to beat $0.103868\text{ eV}$, the scale-transfer hypothesis is falsified.

## 3. Training Contract
- **Dataset**: `nvoid912/pcqm4mv2-ogb-fixed-500k-scnet-v1` (500,000 train rows, 50,000 internal development rows).
- **Architecture**: `GPTransNoisyNodesPairNorm` ($5,277,400$ parameters).
- **Optimizer**: AdamW, Cosine decay schedule over 60 epochs ($4\times 10^{-4} \rightarrow 1\times 10^{-6}$).
- **Batch Size**: 128, drop last.
- **Precision**: FP32 deterministic algorithms.
