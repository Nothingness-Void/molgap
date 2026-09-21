# Final Decision: PCQM GPTrans-T Noisy Nodes 100K

- **Trajectory ID**: `TB-gptrans-noisy-nodes-100k-s42`
- **Contract**: `MOLGAP-COMMON-V5-FINAL`
- **State**: `POSITIVE_UNDER_CONTRACT`
- **Benchmark**: `pcqm4mv2-ogb-fixed-100k-gap-v4`
- **Hypothesis**: Auxiliary node-level denoising loss (predicting clean atomic numbers with alpha=0.1 from corrupted atom embeddings with noise_std=0.15) prevents over-smoothing and lowers development MAE below the 0.156627 eV baseline.
- **Reference**: `pcqm-gptrans-t-100k-v4-reference` (MAE: `0.156627 eV`)
- **Observed Result**: MAE: `0.151788 eV` (at Epoch 59)
- **Material Gain**: `0.004840 eV` improvement over reference (exceeding the required `0.003 eV` falsifier threshold, threshold was `< 0.153627 eV`).
- **Inference Overhead**: Exactly 0 parameters / 0 compute overhead in evaluation mode (noise disabled, aux head bypassed).
- **Physical Cost**: 1.7625 Tesla T4 device hours (2.054 wall hours).
- **Final Determination**: `POSITIVE_UNDER_CONTRACT`. Noisy Nodes successfully regularizes the GPTrans-T architecture on PCQM4Mv2.
- **Next Allowed Action**: Advance Noisy Nodes mechanism to 500K scale-transfer test.
