# Deterministic BRICS Fragment-State GPS9 Screen

**Protocol frozen: 2026-09-09**

This is one Track C architecture question.  It tests whether an explicit
chemistry-defined fragment stream improves the accepted OGB EdgeState GPS9
backbone without changing the target, source, split, optimizer, precision, or
physical batch size.  It is a screen, not a production change or a PCQM
leaderboard run.

## Paired arms

| Arm | Definition |
|---|---|
| Control | Fresh `OGBEdgeStateStructuralGPSWrapper`: 192 hidden channels, 9 GPS layers, 4 heads, RWSE16, persistent edge state 64, mean pooling |
| Candidate | The same control plus BRICS heavy-atom fragment state, 32 fragment channels, directed cut-bond edges, and low-rank fragment-to-atom exchange at layers 3/6/9 |

The candidate uses canonical RDKit SMILES only to define deterministic BRICS
cuts.  Fragment features are structure-only: normalized size, atomic-number
statistics, hetero/aromatic/ring fractions, mean degree, and cut degree.  No
target, prediction, pretrained weight, warm start, auxiliary loss, geometry,
external data, fusion, router, or official-role data enters either arm.

## Data contract

- Source: the checked-in QM9 v3 processed tensor and its paired `gdb9.sdf`.
- Canonicalization: remove hydrogens, canonical isomeric SMILES, then rebuild
  the OGB 9-channel atom and 3-channel bond representation.
- Structural input: OGB graph plus RWSE16.
- Roles materialized: exactly 30,000 train and 3,000 validation graphs.
- The canonical-valid source pool is recorded for identity, but no QM9 test
  graph is materialized or read.
- Cache shards are 2,000 graphs, atom/fragment IDs are checked after PyG
  batching, and every shard is independently hashed.

## Training contract

- Seed: 42 for initialization, split, NumPy, Python, Torch, and loaders.
- Target: direct QM9 Gap in eV, normalized by train mean and standard deviation
  for the loss and reported in raw eV for validation.
- Optimizer: AdamW, learning rate `4e-4`, weight decay `1e-5`.
- Schedule: cosine annealing to `1e-6`, maximum 40 epochs, patience 8.
- Precision: FP32; gradient norm clipped to 1.0.
- Physical batch: 128 for both arms, on one SCNet DCU.
- Both arms checkpoint atomically after every epoch with model, optimizer,
  scheduler, RNG state, trace, best model, and a completion manifest.

## Acceptance and gate

The CPU cache must pass independent acceptance before any DCU allocation.  The
DCU preflight must run a real batch of 128 graphs through forward and backward
for both arms with finite loss and gradients.  The training result is eligible
for a PCQM-100K transfer nomination only when:

1. candidate validation Gap MAE is lower than the paired control; and
2. the paired improvement is at least `0.001 eV`.

A smaller win is weak evidence and closes this exact implementation.  A pass
does not authorize a PCQM run automatically; transfer requires a separately
frozen Track B protocol and budget decision.  Official validation, test-dev,
and any sealed role remain unread.

## Remote layout

The Xi'an SCNet payload is kept below the account home directory:

```text
$HOME/molgap-results/trackc-qm9-fragment-state-<commit>/
  cache/                  # immutable train/validation cache
  preflight/preflight.json
  output/{control,fragment_state}/
  logs/
```

The source code is uploaded as a separate read-only copy.  The queue sequence
is `accept_cache_xian.slurm` -> `preflight_xian.slurm` -> `train_xian.slurm`.
Existing 96-batch jobs are not modified by this experiment.
