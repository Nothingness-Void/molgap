# Kunshan persistent-vector seed-42 protocol

Protocol date: 2026-09-05. Plan: kunshan_discovery_plan.md, K0/K1.

## Identity and mechanism

Control: ogb_distance_angle_triangle_edge_state_graph_state9 (3,665,809 parameters).
Candidate: ogb_distance_angle_vector_state_triangle_edge_state_graph_state9.
Expected candidate count: 3,696,209 (30,400 added); verify on the remote device.
Keep OGB categorical atom/bond inputs, RWSE16, node192, bond64, wedge16,
GraphState64, nine local blocks, and graph exchange at blocks 3/6/9.

Add a 16-channel Cartesian order-1 vector per atom, initialized to zero and
updated by one shared cell after blocks 2/4/6/8. Messages travel only along
the original real bonds. Scalar coefficients may depend on atom/bond states
and distance bases; vector channels use bias-free linear channel mixing,
directed bond displacements, invariant norm/dot-product contractions and scalar gates.
Mask failed geometries. The scalar return projection is zero initialized.
The final prediction is invariant to translation, rotation and reflection.

There is no new conformer, contact graph, high-order tensor, force label,
geometry teacher, target residual or prediction fusion. This tests persistent
orientation propagation, not the previously closed one-shot body-order feature.

## Frozen data and training

Geometry aggregate SHA-256:
3e4206fd239942ab79f9c4978cd4334f6025ba14e8f1cd1df78c20060a0d1d22.
100,000 train and 10,000 internal-validation graphs; all official evaluation
roles stay sealed. Seed42, FP32, batch48, workers0, dropout0.1, AdamW1.6e-4,
weight_decay1e-6, normalized L1 using train-only mean/std, cosine eta_min1e-6,
40 epochs, patience8. New output directory; no 3-epoch probe continuation.

Both models are independently initialized under seed42, preserving common
initial parameters. A separate seed42 DataLoader generator fixes shuffle.
Run sequentially on the same single Kunshan DCU and software environment.
Use the existing reusable train_one implementation; disabling pinned memory
is the declared DTK transfer adaptation, identical for both models.

## Gates before training

Local checks execute AST/static contract checks only. Remote bounded tests
must check common initialization, zero-return identity, finite forward/backward,
nonzero new-path gradients, nontrivial rotation/reflection/translation and
node-order invariance, invalid-geometry masks, empty-edge graphs and batch
isolation. Symmetry tests must enable the learned scalar return to avoid
vacuous agreement with the control. Exact model parameter counts are emitted
remotely and frozen in the launch/preflight evidence, with a strict 4M cap.
If preflight fails, neither long training is launched.

## Durability, cap and acceptance

One 12-hour Slurm allocation covers the complete paired experiment; reserve
30 minutes for exit/artifacts and use an 11.5-hour runner bound. Checkpoint
model, optimizer, cosine scheduler, best/stale epochs, Python/NumPy/Torch/DCU
and DataLoader RNG state atomically after every epoch. Save best model and
per-epoch trace separately. A same-run resume requires identical source,
candidate, cache and RNG state. Infrastructure retries count toward the
12-hour paired allocation cap and the 32-hour discovery cap.

Publish aligned internal-validation prediction CSV plus tensor payload,
per-file hashes, train normalization, runtime versions, parameters and memory.
No-model acceptance recomputes CSV MAE and verifies role/source/cache identities,
row/target alignment, complete 40-epoch or patience8 stopping, and hashes.
Incomplete runs cannot select a winner. Candidate promotion criteria are owned
by kunshan_discovery_plan.md; no automatic confirmation seeds are allowed.
