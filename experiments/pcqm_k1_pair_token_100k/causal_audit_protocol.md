# Frozen PairToken causal audit

The accepted seed-42 checkpoint is evaluated without training on the same
50,000-row development role. Five counterfactuals isolate what the trained
relation branch uses:

- disable the complete PairToken residual;
- replace learned pair selection with a uniform all-pair average;
- retain only diagonal/self pairs;
- retain only off-diagonal/cross-node pairs;
- remove per-pair channel normalization.

The trained candidate is the baseline and must reproduce its saved payload.
Weights, K1 state, data order, FP32 physical batch 128, and sealed-role access
remain unchanged. These interventions diagnose necessity inside the trained
model; they do not estimate how a separately retrained ablation would perform.

The audit may release one new architecture question only when a mechanism has
a coherent overall and topology-stratum effect. It cannot release another
seed, scale-up, or official-role access.

