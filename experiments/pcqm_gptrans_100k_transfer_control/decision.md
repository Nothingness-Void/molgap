# Local 100K GPTrans throughput preflight

The September 25, 2026 train-role-only preflight passed cache SHA checks and
ran the existing GPTrans V4 forward, backward, FP32 AdamW, gradient clipping,
and EMA path on one RTX 5060. It did not train or evaluate a scientific model.

The 200-step sample averaged 0.096812 s/step with physical batch 128. A
50-batch train-role forward sample averaged 0.037428 s/batch. Extrapolating
the frozen 60-epoch, 46,860-step contract gives 1.260 h training, or 1.504 h
including a *synthetic* 50K forward pass each epoch. Checkpointing, epoch
startup, EMA state swaps, role-specific evaluation work, and sustained thermal
behavior were not measured. Peak CUDA reserved memory was 0.885 GB of 8.518
GB; this is not a guarantee for every variant.

The separate accepted Kaggle T4 Noisy Nodes and Joint Noisy+PairNorm 100K
records report 2.054 and 2.0382 h wall time respectively. This is a cross-
hardware, cross-run operational comparison, not a strictly paired scientific
result. The local single-model practical planning range is about 1.6-2.0 h,
subject to full-run confirmation. A two-model local pair is sequential on one
GPU, about 3.2-4.0 h; two available Kaggle slots can finish parallel jobs
sooner in elapsed wall time, but queueing is unknown and uses T4 quota.

The input was the first 100K train rows of the accepted 500K graph cache,
not an assertion that this graph cache equals the historical 100K scientific
contract. No development, official-validation, test, or protected role was
opened. No full training is released by this result. Before a matched pair,
freeze exact data/init/optimizer/EMA/selection identities and a resumable
cost ceiling under a separate prospective action.

Machine-readable measurement: [preflight_result.json](preflight_result.json).
Kaggle comparison evidence: `experiments/pcqm_gptrans_noisy_nodes_100k/costs/`
and `experiments/pcqm_gptrans_noisy_pair_norm_100k/costs/`.
