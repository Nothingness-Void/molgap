# PCQM local-state seed-42 screens

Protocol date: 2026-09-06. These are the two remaining random-initialized,
literature-backed local-state questions from `architecture_route_audit.md`.
They are independent paired screens, not a stack and not confirmation seeds.

## Frozen comparison

Both jobs train a fresh
`ogb_distance_angle_triangle_edge_state_graph_state9` control beside exactly
one candidate. They reuse the accepted ETKDG geometry cache and the official-
train-derived 100,000/10,000 split. Seed 42, FP32, batch 48, AdamW `1.6e-4`,
weight decay `1e-6`, normalized L1, cosine decay, 40 epochs and patience 8 are
unchanged. Official validation and test-dev remain sealed.

Each Kaggle job explicitly requests T4x2. GPU 0 owns the fresh control and GPU
1 owns the candidate; processes have independent RNG, model, optimizer,
checkpoint and output directories. The job has a 14,400-second internal
budget and atomic epoch checkpoints. No local model execution is allowed.

## Account 1: PNA-style neighborhood statistics

Candidate:
`ogb_distance_angle_pna_statistics_triangle_edge_state_graph_state9`.
It retains the complete GraphState9 encoder and adds one shared real-bond
neighborhood path after every block. Edge-conditioned messages are summarized
with mean, maximum, minimum, standard deviation and log degree, then returned
through a rank-64 gated projection. The return projection is zero initialized,
so the initial function and every shared parameter match the control. Expected
parameters: 3,724,755 versus 3,665,809 for the control.

## Account 2: gated persistent-edge retention

Candidate:
`ogb_distance_angle_retention_triangle_edge_state_graph_state9`. It leaves each
baseline edge update intact and adds a rank-32 gate over the previous and
proposed bond states. A zero-initialized value projection makes the initial
function identical while permitting training to retain or correct bond memory
instead of always accepting the additive update. Expected parameters:
3,743,281 versus 3,665,809 for the control.

## Decision boundary

Mechanical acceptance requires exact source/cache hashes, T4x2 isolation,
finite preflight gradients, expected parameter counts, aligned 10,000-row
validation payloads, artifact hashes and sealed-role flags. Scientific ranking
uses only the fresh paired validation MAE, throughput and peak memory. A route
must improve by at least 0.001 eV, run at no more than 1.5x the paired control
time and retain at least 15% device memory to become a promising seed-42
candidate. Completion never auto-submits seeds 43/44 or full-data training.

