# Remote Handoff - K1 Normalization Ablation 100K Dual-Arm Comparison

## Execution Identity
- Platform: Kaggle1
- Accelerator: `NvidiaTeslaT4` (2 GPUs: `cuda:0` and `cuda:1`)
- Kernel: `nothingnessvoid/molgap-k1-norm-ablation-100k-s42`
- Source Dataset: `nothingnessvoid/molgap-k1-norm-ablation-100k-s42-src`
- Fixed Dataset: `nothingnessvoid/pcqm4mv2-ogb-fixed-100k-v1`
- Manifest SHA256: `1b0e8fd579ab1cb86c02e833e7ad284b4af7582b059f912a77853fdccf3ede6d`

## Dual Arms
- `cuda:0`: `k1_single_bn` (single forward with standard BatchNorm, seed 42)
- `cuda:1`: `k1_ada_rmsnorm` (single forward with AdaRMSNorm, seed 42)

## Offline Operating Protocol
- Desktop submits kernel and retrieves initial launch receipt.
- Remote execution is completely self-contained and durable.
- Checkpoints, predictions, and training logs write atomically to Kaggle working directory.
- Reconciliation occurs when desktop returns online; no server takeover or live heartbeat bridge.
