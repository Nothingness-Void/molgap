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

## Paired training and frozen-cohort diagnosis

The separately frozen v2 pair completed on one local RTX 5060. Both arms used
the same source archive, seed-42 core initialization, 100K rows, 50K internal
development rows, 60 epochs, 46,860 optimizer steps, and EMA selection at
epoch 59. The v1 partial reference attempt is excluded; see `correction.md`.
`acceptance.json` records the verified artifact hashes and timing scopes.

On the aligned 100K internal development role, the reference reached
0.155891 eV and the joint Noisy Nodes + Pair Update Norm candidate reached
0.149705 eV: a paired gain of 0.006185 eV. A 1,000-draw row-bootstrap 95%
interval for the gain is [0.005287, 0.007153] eV. This does not account for
training-seed variation. The candidate was worse at epoch 20 by 0.019159 eV,
ahead at epoch 32 by 0.013806 eV, then its lead contracted to 0.006185 eV
by epoch 59. The single-seed 100K effect is real under this local contract,
but the curve warns against treating an intermediate lead as a terminal gain.

Without retraining, both selected 100K checkpoints predicted the previously
used 500K internal development cohort. The reference reached 0.156503 eV,
the candidate 0.150711 eV, for a paired gain of 0.005792 eV (row-bootstrap
interval [0.004801, 0.006675] eV). Changing only the evaluation cohort
therefore preserved most of the local gain. Cohort composition alone is not a
sufficient explanation for the historical gain contraction after 500K
training. The remaining possibilities include training-set size, exposure and
optimization horizon, selection differences, and run/seed variation; this
experiment does not distinguish them causally.

The historical Kaggle joint packager invokes `molgap.noisy_nodes` without an
initial-state argument. The local v2 pair deliberately shares a frozen core.
The historical +0.009382 eV 100K gain and +0.002056 eV 500K gain are useful
context but are not matched replicas of this initialization contract. No
official validation, test-dev, or challenge role was used. This diagnostic
does not release 500K training or change the recommended model. The 500K
development cohort has prior selection history and is not a new holdout.
