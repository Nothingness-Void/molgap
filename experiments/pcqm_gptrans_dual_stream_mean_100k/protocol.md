# Protocol: GPTrans Dual-Stream Mean Readout (DSMR) 100K

## 1. Scientific Objective
Evaluate whether the +10.49 meV gain of Dual-Stream Readout (DSAR) stems purely from **structural all-atom access** or strictly requires **dynamic attention weighting**. By replacing learned attentive atom pooling (`self.pool_gate`, 33K parameters) with parameter-free masked mean pooling (`dual_stream_mean`), we disentangle architectural coverage from chemical attention gating on the frozen 100K PCQM4Mv2 contract.

## 2. Experimental Arms
- **Arm A (DSMR + Pair Norm)**: GPTrans-T with Pair Update LayerNorm + Dual-Stream Mean Readout (5,312,353 parameters).
- **Arm B (DSMR + Noisy Nodes + Pair Norm)**: GPTrans-T with Noisy Nodes ($\alpha=0.1, \sigma=0.15$) + Pair Update LayerNorm + Dual-Stream Mean Readout (5,342,936 parameters).

## 3. Fixed Environment & Resource Discipline
- **Platform**: Kaggle Tesla T4 (single dual-GPU job, isolated worker processes on GPU 0 and GPU 1).
- **Dataset**: `nothingnessvoid/pcqm4mv2-ogb-fixed-100k-v1` (fixed 100,000 train rows, 50,000 development rows).
- **Epochs**: 60 epochs per arm (46,860 optimizer steps, 5,998,080 presentations).
- **Batch Size**: 128 (physical, deterministic FP32, TF32 disabled).
- **Optimizer**: AdamW, lr=4e-4, Cosine decay to 1e-6, weight decay=1e-5, grad clip=1.0.
- **Expected Duration**: ~1.4 wall hours on Dual T4 (~2.8 device hours).

## 4. Frozen Acceptance Threshold & Comparators
- **Baseline Reference**: `pcqm-gptrans-t-100k-v4-reference` (`0.156627 eV`).
- **Primary Falsifier**: $\text{MAE} < 0.153627\text{ eV}$ (at least `0.003 eV` gain over baseline).
- **Attentive Readout Comparators**:
  - Attentive Arm A (`TB-gptrans-dsar-pair-norm-100k-s42`): `0.146192 eV`
  - Attentive Arm B (`TB-gptrans-dsar-noisy-pair-norm-100k-s42`): `0.146138 eV`
