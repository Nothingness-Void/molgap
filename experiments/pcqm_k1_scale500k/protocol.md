# K1 PCQM 500K scale bridge

## Question

Does the frozen one-slot Neural-Atom K1 architecture retain its paired
advantage over fresh full GPS when only the training-role cardinality changes
from 100K to 500K?

## Release boundary

The user authorized this scale bridge on 2026-09-10 and revised its execution
owner on 2026-09-11. After the frozen K1 shadow audit passed, `molgap-server`
was authorized to construct and independently accept the cache, then run the
paired comparison on Kaggle2. This does not authorize full-scale training or
official validation/test-dev; those remain desktop-owned decisions.

Cache construction is a CPU-only job. The T4x2 training task may be submitted
only after the retrieved cache passes the frozen no-model acceptance. Each arm
receives one T4 and an independent model, RNG, optimizer, and checkpoint tree.

## Frozen data roles

- Source: first 3,378,606 official PCQM4Mv2 training rows only.
- Base train: the exact accepted 100K role, SHA-256
  `d08e04ef73090b77963a6959d7efa22d22a4085e3869a0e13086491b8f8c9678`.
- Development: the exact accepted 10K role, unchanged, SHA-256
  `40b210c03789249f89950d7eb9df6de93ec151903f75077cfa64fe0a94eca5b0`.
- Added train: 400K deterministic rows selected from the official-train
  complement with NumPy `default_rng(2026091050)`, excluding the base train,
  unchanged development role, all construction reserves, and the accepted K1
  shadow role.
- Final train: union of the original 100K and added 400K, exactly 500K rows.
- Official validation, test-dev, test-challenge, and shadow labels remain
  unread.

The executable row-identity contract is `molgap.pcqm_k1_scale`. Desktop must
build an immutable pure-2D cache from its output and accept every shard and the
aggregate identity before accelerator work. The existing prefix-500K/50K cache
must not be substituted because it changes the development role and cannot
measure retention of the accepted 100K effect.

## Frozen architecture

- Candidate identity: `neural_atom_k1` from source commit
  `47f99cf9da7fee306f5165175b4020c6c4aa9fb3`.
- OGB nine-field atom encoder, OGB three-field real-bond encoder, RWSE16.
- Node width 192, nine local persistent-EdgeState blocks, EdgeState width 64.
- One 64-channel Neural-Atom slot at layers 3, 6, and 9.
- No dense atom-to-atom global attention in K1.
- Mean pooling and one direct scalar Gap head.
- Expected inference parameters: 3,658,817.
- Pure 2D; no coordinates, geometry targets, pretraining, warm start,
  distillation, residual target, prediction fusion, or teacher.

## Paired training contract

Two fresh arms run in one task/platform on the same accelerator class:

1. `full_gps`: accepted 192-wide nine-layer EdgeState GPS control;
2. `neural_atom_k1`: frozen candidate above.

`full_gps` means fresh full-attention GPS, not a checkpoint trained on the
complete PCQM dataset. Both arms start from seed-42 initialization on this
500K role; no historical checkpoint is loaded.

Both use seed 42, FP32 throughout, physical batch 128 per model/device, no
accumulation, fused AdamW with learning rate `4e-4`, weight
decay `1e-5`, clipping 1.0, cosine 40 epochs to `1e-6`, two pinned-memory
loader workers per arm, identical deterministic row order, and exactly 40
direct-Gap passes. The optimizer schedule is indexed by optimizer step over
500K and is identical between arms. No parameter, width, depth, feature,
dropout, target, or validation change is allowed. Pinned non-blocking transfer
and two loader workers per arm are throughput-only settings. The user withdrew
the unlaunched FP16 AMP amendment on 2026-09-11 so this bridge remains directly
comparable with the established FP32 screens.

Every epoch saves an atomic resumable checkpoint, trace, best model, and
aligned development predictions. A real batch-128 forward/backward preflight
and at least 15% memory reserve are mandatory.

## Decision gate

The 100K paired K1 gain was `0.009461689 eV`. At 500K, K1 advances only if:

- its paired development MAE is lower than fresh full GPS;
- the paired row-bootstrap 95% interval excludes zero in K1's favor;
- the gain is at least `0.0047308445 eV`, retaining at least half of the 100K
  gain;
- all identity, role, finite-value, checkpoint, and hash checks pass.

Failure closes K1 scale-up without changing architecture or trying another
seed. Passing authorizes a separate desktop full-run budget decision; it does not open
official validation or test-dev.
