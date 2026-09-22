# Final Decision: PCQM GPTrans-T Dual-Stream Attentive Readout (DSAR) + Pair Norm 100K (Arm A)

- **Trajectory ID**: `TB-gptrans-dsar-pair-norm-100k-s42`
- **Contract**: `MOLGAP-COMMON-V5-FINAL`
- **State**: `POSITIVE_UNDER_CONTRACT`
- **Benchmark**: `pcqm-fixed100k-dev50k-matched60-v4`
- **Hypothesis**: Dual-Stream Attentive Readout (DSAR) pooling 7,680-dim real atomic features alongside the virtual node token with Pair Update LayerNorm operates constructively to reduce error below both the GPTrans baseline (0.156627 eV) and the previous project record (0.147245 eV).
- **Reference**: `pcqm-gptrans-t-100k-v4-reference` (MAE: `0.156627 eV`)
- **Previous Project Record**: `pcqm-gptrans-noisy-pair-norm-100k-s42` (MAE: `0.147245 eV`)
- **Observed Result**: MAE: `0.146192 eV` (at Epoch 59)
- **Material Gain vs Baseline**: `0.010435 eV` (+10.44 meV) improvement over baseline reference.
- **Material Gain vs Previous Record**: `0.001053 eV` (+1.05 meV) improvement over previous all-time project record.
- **Falsifier Threshold**: `< 0.153627 eV` (cleared with huge surplus).
- **Physical Cost**: 2.436 Tesla T4 device hours.
- **Final Determination**: `POSITIVE_UNDER_CONTRACT`. Dual-Stream Attentive Readout (DSAR) with Pair Update LayerNorm decisively outperforms virtual-token-only readout, establishing an all-time record on 100K PCQM4Mv2 without needing auxiliary node perturbation.
- **Next Allowed Action**: Nominate DSAR as candidate architecture for 500K scale-up and scale-transfer evaluation.
