# Final Decision: PCQM GPTrans-T Dual-Stream Attentive Readout (DSAR) + Noisy Nodes + Pair Norm 100K (Arm B)

- **Trajectory ID**: `TB-gptrans-dsar-noisy-pair-norm-100k-s42`
- **Contract**: `MOLGAP-COMMON-V5-FINAL`
- **State**: `POSITIVE_UNDER_CONTRACT`
- **Benchmark**: `pcqm-fixed100k-dev50k-matched60-v4`
- **Hypothesis**: The joint integration of Dual-Stream Attentive Readout (DSAR), auxiliary Noisy Nodes atom denoising ($\alpha=0.1, \sigma=0.15$), and Pair Update LayerNorm operates constructively to establish a new state of the art below the previous project record ($0.147245\text{ eV}$) and baseline ($0.156627\text{ eV}$).
- **Reference**: `pcqm-gptrans-t-100k-v4-reference` (MAE: `0.156627 eV`)
- **Previous Project Record**: `pcqm-gptrans-noisy-pair-norm-100k-s42` (MAE: `0.147245 eV`)
- **Observed Result**: MAE: `0.146138 eV` (at Epoch 59)
- **Material Gain vs Baseline**: `0.010489 eV` (+10.49 meV) improvement over baseline reference.
- **Material Gain vs Previous Record**: `0.001107 eV` (+1.11 meV) improvement over previous all-time project record.
- **Falsifier Threshold**: `< 0.153627 eV` (cleared with massive surplus).
- **Physical Cost**: 2.515 Tesla T4 device hours.
- **Final Determination**: `POSITIVE_UNDER_CONTRACT`. All-time MolGap 100K record on PCQM4Mv2 (`0.146138 eV`). DSAR attentive atom pooling combined with Noisy Nodes representation regularization achieves the lowest error in the history of the benchmark.
- **Next Allowed Action**: Advance joint DSAR + Noisy Nodes + Pair Norm architecture to 500K scale-transfer testing and nomination.
