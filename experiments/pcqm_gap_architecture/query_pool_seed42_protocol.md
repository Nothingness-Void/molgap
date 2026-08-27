# Learned-query Pooling GPS9 Seed-42 Protocol

## OGB compliance

- Task: direct graph regression of the PCQM4Mv2 HOMO-LUMO Gap in eV.
- Selection metric: lower mean absolute error on the frozen internal
  official-train-derived validation role.
- Inputs: official OGB categorical 2D graph fields plus topology-derived RWSE16.
- No external training data, text data, official test-dev access, auxiliary
  labels, 3D coordinates, pretrained weights, residual targets, or prediction
  fusion.
- The architecture must retain a credible raw-SMILES-to-prediction path below
  the official four-hour single-GPU plus single-CPU inference budget.

## Single changed mechanism

`ogb_query_pool_structural_gps9` retains the accepted Structural GPS9 atom,
bond, RWSE, nine-layer local/global encoder, and scalar head. It replaces
uniform mean pooling with four learned graph queries. The queries cross-attend
once to the final node set, pass through a bounded feed-forward residual, and
are averaged before the scalar head.

This is neither a persistent EdgeState nor the closed recurrent graph-token,
JK-readout, shortest-path, directed-edge, or multihop branch.

## Frozen run contract

- Runtime source commit:
  `1d67bd364113f05992934242b334b176c785601f`.
- Cache SHA-256:
  `eb7c843e33f430ac755bc575d80153aba87677cea1ad5bb0dcf73cca906e2c21`.
- Seed 42, FP32, batch 48, AdamW, learning rate `1.6e-4`, weight decay
  `1e-6`, at most 40 epochs, patience 8.
- Parameter ceiling: 5.2M.
- One candidate and one GPU task; no retraining of the frozen comparator.
- Strict advancement threshold: internal-validation Gap MAE below
  `0.13798263211250306 eV`.
- A failure closes this mechanism without query-count, width, seed, or schedule
  retry. A success freezes the candidate for seeds 43/44 but does not submit
  them automatically.
