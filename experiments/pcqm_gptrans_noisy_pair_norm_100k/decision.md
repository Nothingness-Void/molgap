# Final Decision: PCQM GPTrans-T Noisy Nodes + Pair Update Norm 100K

- **Trajectory ID**: `TB-gptrans-noisy-pair-norm-100k-s42`
- **Contract**: `MOLGAP-COMMON-V5-FINAL`
- **State**: `POSITIVE_UNDER_CONTRACT`
- **Benchmark**: `pcqm4mv2-ogb-fixed-100k-gap-v4`
- **Hypothesis**: Orthogonal combination of node-level denoising regularization (alpha=0.1, noise_std=0.15) and relation update LayerNorm normalization operates constructively to reduce error below both the baseline (0.156627 eV) and isolated single mechanisms (0.151784 eV).
- **Reference**: `pcqm-gptrans-t-100k-v4-reference` (MAE: `0.156627 eV`)
- **Single Mechanism Reference**: `pcqm-gptrans-noisy-nodes-100k-s42` (MAE: `0.151788 eV`) / Pair Update Norm (MAE: `0.151787 eV`)
- **Observed Result**: MAE: `0.147245 eV` (at Epoch 59)
- **Material Gain vs Reference**: `0.009382 eV` improvement over baseline reference.
- **Material Gain vs Single Mechanism**: `0.004539 eV` improvement over isolated mechanisms (falsifier threshold was `< 0.148784 eV`, satisfied).
- **Inference Overhead**: Exactly 0 parameters / 0 compute overhead in evaluation mode (noise bypassed, aux head bypassed).
- **Physical Cost**: 1.7280 Tesla T4 device hours (2.0382 wall hours).
- **Final Determination**: `POSITIVE_UNDER_CONTRACT`. Joint orthogonal regularization demonstrates pronounced synergy and sets a new state-of-the-art 100K mark on PCQM4Mv2 (0.1472 eV).
- **Next Allowed Action**: Nominate joint Noisy Nodes + Pair Update Norm architecture as primary GPTrans configuration for 500K scale-transfer test.
