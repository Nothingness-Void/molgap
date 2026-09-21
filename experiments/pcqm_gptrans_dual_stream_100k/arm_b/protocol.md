# Protocol: GPTrans DSAR + Noisy Nodes + Pair Update Norm 100K (Arm B)

## 1. Scientific Objective
Evaluate whether combining **Dual-Stream Attentive Readout (DSAR)** with **Noisy Nodes auxiliary denoising** (alpha=0.1, noise_std=0.15) and **Pair Update LayerNorm** eliminates the readout bottleneck, regularizes atom representations, and breaks through the previous 100K project record (`0.147245 eV`) as well as the baseline (`0.156627 eV`).

## 2. Model Architecture
- **Model Family**: GPTrans-T with Noisy Nodes (`GPTransNoisyNodes`)
- **Variant**: `pair_update_norm` + `readout_mode="dual_stream_attentive"`
- **Parameters**: 5,375,961
- **Auxiliary Loss**: Cross-entropy denoising head on masked atom representations (alpha=0.1, noise_std=0.15)

## 3. Fixed Environment & Training Hyperparameters
- **Platform**: Kaggle Tesla T4 (Worker GPU 1).
- **Run ID**: `nothingnessvoid/molgap-pcqm-gptrans-dsar-noisy-pair-norm-s42`
- **Output Directory**: `/kaggle/working/arm_b`
- **Dataset**: `nothingnessvoid/pcqm4mv2-ogb-fixed-100k-v1` (100K train, 50K development).
- **Epochs**: 60 (46,860 optimizer steps, 5,998,080 presentations).
- **Batch Size**: 128 (physical, FP32 deterministic).
- **Optimizer**: AdamW, lr=4e-4, Cosine decay to 1e-6, weight decay=1e-5, grad clip=1.0.
- **Expected Duration**: ~1.4 wall hours on Tesla T4.

## 4. Frozen Acceptance Threshold
- **Primary Falsifier**: `< 0.153627 eV` (at least `0.003 eV` gain over baseline).
- **Secondary Aspiration**: `< 0.147245 eV` (surpassing prior joint regularized record).
