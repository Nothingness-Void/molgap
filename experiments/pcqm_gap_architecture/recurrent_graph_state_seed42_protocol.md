# Recurrent Graph-State EdgeState Seed-42 Protocol

## Question

Does a compact recurrent molecule state improve the frozen persistent-
EdgeState GPS9 comparator for official-PCQM direct Gap prediction?

The prior QM9 graph-token candidate improved Gap but failed a three-target
average gate because LUMO regressed. This experiment asks a different question:
the official PCQM4Mv2 task predicts Gap only. GPS++ independently supports a
node-edge-global state path for this dataset.

## Frozen architecture

- Candidate key: `ogb_recurrent_graph_state_gps9`.
- Official OGB categorical atom and bond encoders.
- Topology-only RWSE16 input.
- Nine GPS layers, width 192, four attention heads, dropout 0.1.
- Persistent 64-channel real-bond EdgeState retained unchanged.
- One shared 192-channel graph state, updated after every GPS block from the
  previous graph state and mean-pooled nodes through a 16-channel bottleneck.
- The graph state is normalized and broadcast to nodes before every block.
- Update and broadcast output projections are zero-initialized.
- Mean node pooling and one direct scalar Gap head.
- Parameter ceiling 5.2M; the remote preflight owns the measured count.

No checkpoint, pretraining, distillation, prediction fusion, residual target,
HOMO/LUMO auxiliary target, 3D input, text, external data, or new graph feature
is allowed.

## Frozen data and optimization

- Accepted official-train-derived cache aggregate SHA-256:
  `eb7c843e33f430ac755bc575d80153aba87677cea1ad5bb0dcf73cca906e2c21`.
- Exactly 100,000 effective training graphs and 10,000 internal-validation
  graphs; official validation and test-dev remain unread.
- Seed 42, FP32, batch 48, AdamW, learning rate `1.6e-4`, weight decay
  `1e-6`, at most 40 epochs, patience 8.
- Frozen comparator MAE: `0.13798263211250306 eV` from
  `results/seed42_structural_vs_edge_state/decision.md`.
- One Kaggle GPU task, with a nominal four-hour wall budget and atomic
  checkpoint, model, trace, metrics, and validation-payload outputs.

## Gate and stopping

The candidate advances only if its internal-validation Gap MAE is strictly
below the frozen comparator and downloaded outputs pass no-inference
acceptance, including source/cache/contract identities and every artifact hash.
Only then may a separate decision authorize seeds 43/44. Otherwise this graph-
state mechanism closes without a token-width, seed, optimizer, schedule, or
initialization retry.

No result from this run authorizes official validation, test-dev, full-data
training, or molecular-research-server access.
