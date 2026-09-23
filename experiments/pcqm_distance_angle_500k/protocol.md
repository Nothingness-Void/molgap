# Distance-Angle Triangle EdgeState GPS9 500K protocol

## Question

Does `ogb_distance_angle_triangle_edge_state_gps9` retain its accepted 100K
advantage over the current `ogb_edge_state_structural_gps9` baseline when the
training role is expanded to 500K molecules?

## Frozen roles

- Train: source indices `0..499999`, exactly 500,000 rows.
- Development: source indices `500000..549999`, exactly 50,000 rows.
- Cache aggregate SHA-256:
  `676a506c808402bc16a4437cc02286168239a8dd5de4451e992131eddb4f1b20`.
- Node features: official OGB nine-field categorical schema plus RWSE16.
- Candidate-only geometry inputs: ETKDGv3+MMFF94s real-bond distances and
  non-backtracking wedge angle cosines from the same accepted cache.
- Official validation, test-dev, and test-challenge are not read.

## Matched arms

1. Baseline: OGB-rich EdgeState Structural GPS9, 4,771,073 parameters.
2. Candidate: Distance+Angle Triangle EdgeState GPS9, 4,891,057 parameters.

Both arms use width 192, nine layers, four attention heads, EdgeState width
64, RWSE16, mean pooling, one direct Gap head, seed 42, FP32, physical batch 128,
four loader workers, AdamW at `1.6e-4`, weight decay `1e-6`, gradient
clipping 1.0, three warm-up epochs, cosine decay to `1e-6`, and 60 direct-Gap epochs.
They use identical deterministic shard and within-shard order.

No warm start, pretraining, teacher, residual target, prediction fusion,
calibration, target change, early stopping, or architecture-specific optimizer
change is allowed.

## Durability and acceptance

Every epoch atomically writes a resumable last checkpoint and trace. Each new
best epoch atomically writes the model and the complete aligned 50K development
prediction payload. Completion records source commit, configuration, cache
hash, model hashes, parameter count, and sealed-role flags.

A real physical-batch-128 forward/backward preflight for both arms must pass
before either training job is released. Both arms must complete and pass the
same mechanical acceptance before comparison.

## Decision gate

The candidate is nominated for a larger run only if its paired best
development MAE is at least `0.001 eV` lower than the baseline and all
identity, finite-value, checkpoint, prediction-alignment, and hash checks pass.
Otherwise this scale path closes without another seed or full-data run.
