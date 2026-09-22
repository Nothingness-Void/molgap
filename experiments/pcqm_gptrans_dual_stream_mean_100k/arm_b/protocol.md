# Protocol: GPTrans DSMR + Noisy Nodes + Pair Update Norm 100K (Arm B)

## 1. Scientific Objective
Evaluate whether combining parameter-free **Dual-Stream Mean Readout (DSMR)** with **Noisy Nodes auxiliary denoising** ($\alpha=0.1, \sigma=0.15$) and **Pair Update LayerNorm** improves upon the baseline (`0.156627 eV`), and evaluate how it compares against the attentive counterpart (`0.146138 eV`) to separate structural vs attention-weighted gains under representation regularization.

## 2. Model Architecture
- **Model Family**: GPTrans-T with Noisy Nodes (`GPTransNoisyNodes`)
- **Variant**: `pair_update_norm` + `readout_mode="dual_stream_mean"`
- **Parameters**: 5,342,936
- **Auxiliary Loss**: Cross-entropy denoising head on masked atom representations ($\alpha=0.1, \sigma=0.15$)

## 3. Fixed Environment & Training Hyperparameters
- **Platform**: Kaggle Tesla T4 (Worker GPU 1).
- **Run ID**: `nothingnessvoid/molgap-pcqm-gptrans-dual-stream-mean-s42`
- **Output Directory**: `/kaggle/working/arm_b`
- **Dataset**: `nothingnessvoid/pcqm4mv2-ogb-fixed-100k-v1` (100K train, 50K development).
- **Epochs**: 60 (46,860 optimizer steps, 5,998,080 presentations).
- **Batch Size**: 128 (physical, FP32 deterministic).
- **Optimizer**: AdamW, lr=4e-4, Cosine decay to 1e-6, weight decay=1e-5, grad clip=1.0.
- **Expected Duration**: ~1.4 wall hours on Tesla T4.

## 4. Frozen Acceptance Threshold & Comparators
- **Baseline Reference**: `pcqm-gptrans-t-100k-v4-reference` (`0.156627 eV`).
- **Primary Falsifier**: $< 0.153627\text{ eV}$ (at least `0.003 eV` gain over baseline).
- **Attentive Comparator**: `TB-gptrans-dsar-noisy-pair-norm-100k-s42` (`0.146138 eV`).
