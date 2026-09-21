# Protocol: GPTrans DSAR + Pair Update Norm 100K (Arm A)

## 1. Scientific Objective
Evaluate whether upgrading GPTrans's readout to a **Dual-Stream Attentive Readout (DSAR)**—pooling real atomic representations alongside the virtual token with Pair Update LayerNorm—improves upon the frozen GPTrans-T 100K baseline (`0.156627 eV`) without auxiliary node denoising.

## 2. Model Architecture
- **Model Family**: GPTrans-T (`OGBGPTransTiny`)
- **Variant**: `pair_update_norm` + `readout_mode="dual_stream_attentive"`
- **Parameters**: 5,345,378
- **Auxiliary Loss**: None (alpha=0.0)

## 3. Fixed Environment & Training Hyperparameters
- **Platform**: Kaggle Tesla T4 (Worker GPU 0).
- **Run ID**: `nothingnessvoid/molgap-pcqm-gptrans-dsar-pair-norm-s42`
- **Output Directory**: `/kaggle/working/arm_a`
- **Dataset**: `nothingnessvoid/pcqm4mv2-ogb-fixed-100k-v1` (100K train, 50K development).
- **Epochs**: 60 (46,860 optimizer steps, 5,998,080 presentations).
- **Batch Size**: 128 (physical, FP32 deterministic).
- **Optimizer**: AdamW, lr=4e-4, Cosine decay to 1e-6, weight decay=1e-5, grad clip=1.0.
- **Expected Duration**: ~1.4 wall hours on Tesla T4.

## 4. Frozen Acceptance Threshold
- **Primary Falsifier**: `< 0.153627 eV` (at least `0.003 eV` gain over baseline).
