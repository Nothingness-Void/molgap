# Protocol: GPTrans Dual-Stream Attentive Readout 100K

## 1. Scientific Objective
Evaluate whether upgrading GPTrans's readout from a single virtual-token bottleneck (`node[:, 0]`) to a **Dual-Stream Attentive Readout (DSAR)**—which adaptively pools all real atomic representations alongside the virtual token—eliminates representation over-squashing and delivers a material improvement over the immutable GPTrans-T 100K baseline (`0.156627 eV`) and the previous regularized 100K project record (`0.147245 eV`).

## 2. Experimental Arms
- **Arm A (DSAR + Pair Norm)**: GPTrans-T with Pair Update LayerNorm + Dual-Stream Attentive Readout.
- **Arm B (DSAR + Noisy Nodes + Pair Norm)**: GPTrans-T with Noisy Nodes (alpha=0.1, noise_std=0.15) + Pair Update LayerNorm + Dual-Stream Attentive Readout.

## 3. Fixed Environment & Resource Discipline
- **Platform**: Kaggle Tesla T4 (single job, dual isolated worker directories).
- **Dataset**: `nothingnessvoid/pcqm4mv2-ogb-fixed-100k-v1` (fixed 100,000 train rows, 50,000 development rows).
- **Epochs**: 60 epochs per arm (46,860 optimizer steps, 5,998,080 presentations).
- **Batch Size**: 128 (physical, deterministic FP32, TF32 disabled).
- **Optimizer**: AdamW, lr=4e-4, Cosine decay to 1e-6, weight decay=1e-5, grad clip=1.0.
- **Expected Duration**: ~2.5 - 2.8 wall hours on Dual T4 (~5.0 - 5.6 device hours).

## 4. Frozen Acceptance Threshold
- **Primary Falsifier**: `< 0.153627 eV` (at least `0.003 eV` gain over the `0.156627 eV` GPTrans-T baseline).
- **Secondary Aspiration**: Exceeding the `0.147245 eV` regularized record.
