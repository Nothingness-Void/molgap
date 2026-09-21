# GPTrans Noisy Nodes + Pair Update Normalization 100K Protocol

## Frozen Question

Under the fixed `ogb-lsc-pcqm4mv2-gap-internal-100k-v4` contract, do auxiliary Noisy Nodes denoising regularization ($\sigma=0.15, \alpha=0.10$) and parameter-free Relation Update Normalization (`pair_update_norm`) combine constructively to yield compounding gains on GPTrans-T, outperforming both isolated winners (`0.151784 eV` and `0.151788 eV`) and the frozen baseline (`0.156627 eV`)?

## Scientific Hypothesis

- **Observed Deficiency**: The standard GPTrans-T baseline (`0.156627 eV`) suffers from two distinct bottlenecks: representation over-smoothing across deep layers, and uncontrolled variance growth in cumulative pair relation representations.
- **Isolated Findings**:
  - `experiments/pcqm_gptrans_noisy_nodes_100k`: auxiliary node perturbation ($\sigma=0.15$) and atomic reconstruction loss ($\alpha=0.10$) achieved `0.151788 eV` (+0.004840 eV gain).
  - `experiments/pcqm_gptrans_pair_norm_100k`: normalizing relation updates before residual addition achieved `0.151784 eV` (+0.004843 eV gain).
- **Compounding Mechanism**: Because Noisy Nodes operates at the atom/node level and `pair_update_norm` operates at the pairwise relation residual boundary, their operational channels are orthogonal. Combining them should prevent representation collapse while stabilizing pair communication.

## Contract Specifications

- **Contract**: `ogb-lsc-pcqm4mv2-gap-internal-100k-v4` under `MOLGAP-COMMON-V5-FINAL`
- **Data Cache**: `nothingnessvoid/pcqm4mv2-ogb-fixed-100k-v1`
  - Manifest SHA256: `1b0e8fd579ab1cb86c02e833e7ad284b4af7582b059f912a77853fdccf3ede6d`
- **Roles**:
  - Training: source indices `0..99999` (100,000 graphs)
  - Internal Development: source indices `100000..149999` (50,000 graphs)
  - Sealed Roles: official validation, test-dev, test-challenge remain unread
- **Architecture**:
  - Backbone: GPTrans-T (12 blocks, node channels 256, pair channels 32, 8 heads)
  - Relation update norm: `pair = pair + LayerNorm(pair_update)`
  - Noisy Nodes: Gaussian embedding perturbation $\sigma=0.15$ during training, linear reconstruction head predicting atomic numbers (119 classes, loss weight $\alpha=0.10$)
  - Parameter count: `5,277,400`
  - Evaluation mode: 0 inference noise, auxiliary head bypassed, 0 compute overhead
- **Training Hyperparameters**:
  - Precision: FP32 (no TF32, deterministic algorithms enabled)
  - Seed: 42
  - Physical batch size: 128 (drop_last tail rows)
  - Epochs: 60
  - Optimizer: AdamW (lr 1e-3, weight decay 0.05, warmup 4 epochs, cosine to 1e-6, grad clip 1.0)
  - EMA: 0.9999

## Decision Gate & Falsifier

- **Primary Reference**: `pcqm-gptrans-t-100k-v4-reference` (`0.156627 eV`)
- **Single-Mechanism Comparators**:
  - `pair_update_norm` (`0.151784 eV`)
  - `pcqm-gptrans-noisy-nodes-100k-s42` (`0.151788 eV`)
- **Falsification Rule**: If the combined candidate fails to improve upon the single-mechanism comparator (`< 0.151784 eV`) with statistically favorable candidate-minus-reference bootstrap interval, the joint hypothesis is falsified (`NEGATIVE_UNDER_CONTRACT` or `POSITIVE_BELOW_GATE`).
- **Target Goal**: Development MAE $< 0.1500\text{ eV}$ (demonstrating compounding orthogonal gain).
