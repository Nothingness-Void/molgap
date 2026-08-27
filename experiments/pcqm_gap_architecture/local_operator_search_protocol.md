# Local-Operator Search Seed-42 Protocol

## Question

Can a materially different edge-aware local message-passing operator improve
the frozen persistent-EdgeState comparator while preserving the official
PCQM4Mv2 pure-2D, Gap-only path and bounded inference cost?

## Frozen common contract

- Runtime source commit:
  `bfaffe332d4c89dc041669679d6fb066e01bce1f`.
- Official OGB categorical 2D atom and bond fields plus topology-only RWSE16.
- Accepted cache aggregate SHA-256:
  `eb7c843e33f430ac755bc575d80153aba87677cea1ad5bb0dcf73cca906e2c21`.
- Seed 42, FP32, batch 48, AdamW, learning rate `1.6e-4`, weight decay
  `1e-6`, at most 40 epochs, patience 8.
- Nine GPS layers, width 192, four global-attention heads, mean pooling, and
  one direct Gap head.
- Parameter ceiling 5.2M and one GPU task at a time.
- No external data, text, official validation, test-dev, 3D, auxiliary targets,
  residual targets, checkpoints, pretraining, or prediction fusion.
- Strict threshold: internal-validation Gap MAE below
  `0.13798263211250306 eV`.

## Predeclared sequence

1. `ogb_gated_local_gps9`: ResGatedGraphConv local branch.
2. `ogb_edge_attention_local_gps9`: bond-conditioned TransformerConv local
   attention.
3. `ogb_gen_local_gps9`: GENConv softmax generalized aggregation.
4. `ogb_gatv2_local_gps9`: optional bond-conditioned GATv2 fallback.

The first three run sequentially under one four-hour search budget. The fourth
runs only if the remaining budget exceeds the slowest completed candidate plus
a ten-minute safety buffer. No candidate may be retried by seed, width,
operator settings, or optimizer changes.

## Evidence and stopping

Every candidate writes an atomic checkpoint, best model, trace, validation
payload, metrics, and hashes. The downloaded outputs undergo no-inference
acceptance. A strict winner may proceed only to a separately authorized
seed-43/44 gate. Otherwise all completed mechanisms close and no full-data or
molecular-research-server work starts.
