# K1 PCQM 500K SCNet-matched benchmark

## Question

On the exact accepted SCNet 500K/50K source-index roles, does the frozen
one-slot Neural-Atom K1 architecture outperform a fresh full-GPS control?

## Release boundary

The user authorized this bounded benchmark for `molgap-server` on Kaggle2.
It does not authorize full-scale training or official validation/test-dev;
those remain desktop-owned decisions. Cache construction is CPU-only and must
pass no-model acceptance before one paired T4x2 training task is submitted.

## Frozen data roles

- Source: official PCQM4Mv2 training rows only.
- Train: source indices `0..499999`, exactly 500,000 rows, SHA-256
  `a9c8b2b698c67f30348c6edbccff00eb9e2c06b064ee4d1a11e607f9531a0f8e`.
- Development: source indices `500000..549999`, exactly 50,000 rows, SHA-256
  `9ae885e5e74d82820d83758942eaf3e6ae2a7dcefbc1f0f4c174ce94c6786bb9`.
- These row identities exactly match the accepted SCNet ESGPS6-304 and
  GPTrans-T screens. Their accepted rich-cache reference is
  `676a506c808402bc16a4437cc02286168239a8dd5de4451e992131eddb4f1b20`.
- Kaggle rebuilds only the pure-2D OGB graph and RWSE16 views needed by K1;
  equality with SCNet is a row/target-role claim, not a byte-identical cache
  claim.
- No replacement or skipped row is allowed. Official validation, test-dev,
  test-challenge, and the consumed K1 shadow role remain unread.

## Frozen architecture

- Candidate identity: `neural_atom_k1` from source commit
  `47f99cf9da7fee306f5165175b4020c6c4aa9fb3`.
- OGB nine-field atom encoder, OGB three-field real-bond encoder, RWSE16.
- Node width 192, nine local persistent-EdgeState blocks, EdgeState width 64.
- One 64-channel Neural-Atom slot at layers 3, 6, and 9.
- No dense atom-to-atom global attention in K1.
- Mean pooling and one direct scalar Gap head; 3,658,817 parameters.
- Pure 2D; no geometry, pretraining, warm start, teacher, residual target, or
  prediction fusion.

## Paired training contract

Two fresh arms run concurrently, one per T4:

1. `full_gps`: 192-wide nine-layer persistent-EdgeState full-attention GPS;
2. `neural_atom_k1`: the frozen candidate above.

Both use seed 42, FP32 throughout, physical batch 128 per device, no
accumulation, fused AdamW at `4e-4`, weight decay `1e-5`, clipping 1.0, and
cosine 40 epochs to `1e-6`. Both use identical deterministic row order, two
pinned-memory loader workers, and independent RNG/optimizer/checkpoint trees.
Pinned transfer and loader workers are execution settings only. The user
withdrew the unlaunched FP16 AMP amendment before GPU submission.

Every epoch saves an atomic checkpoint, trace, best model, and aligned 50K
development predictions. A real batch-128 FP32 forward/backward preflight and
at least 15% memory reserve are mandatory.

SCNet results share data roles, seed, FP32, and physical batch, but some use a
different architecture-specific schedule. Therefore K1-versus-full-GPS is the
causal paired claim; cross-platform SCNet values are contextual comparisons.

## Decision gate

K1 advances only if its paired development MAE is at least `0.001 eV` lower
than fresh full GPS, the paired row-bootstrap 95% interval excludes zero in
K1's favor, and all identity, finite-value, checkpoint, and hash checks pass.
Failure closes this K1 scale benchmark without another seed. Passing authorizes
only a separate desktop full-run budget decision.
