# Protocol: K1 MoSE replacement on fixed PCQM 100K

## Question

Does replacing RWSE16 with 31 rooted motif-homomorphism counts improve the
frozen K1-v4 architecture under the exact V5 PCQM-100K screen?

The hypothesis targets an information deficit, not a capacity deficit.  Prior
K1 variants that added slot, edge, readout, local-block or pair-processing
capacity are closed.  MoSE instead supplies deterministic topology information
that is strictly distinct from RWSE in the cited ICLR 2025 analysis.

## Single change

- Reference: immutable `neural_atom_k1_v4`, 3,658,817 parameters.
- Candidate: `neural_atom_k1_mose`.
- Preserve the nine local persistent real-bond EdgeState blocks and K1 slot
  exchanges at layers 3, 6 and 9.
- Replace, never concatenate, RWSE16 with the official All5 pattern order:
  30 rooted connected patterns on 2--5 nodes plus rooted C6.
- Apply deterministic `log1p` to nonnegative counts before the otherwise
  unchanged two-linear-layer structural encoder.
- No atom/bond labels, targets, geometry, teacher, external data or protected
  role enters count construction.

The candidate adds only 2,880 weights from the wider first structural linear
map.  No width, depth, optimizer, schedule, loss, target transform or exposure
changes.

## V5 screen

- Data: `nothingnessvoid/pcqm4mv2-ogb-fixed-100k-v1`.
- Train/development: fixed official-train-derived 100,000 / 50,000 roles.
- Seed 42, strict FP32/no TF32, physical batch 128, `drop_last`.
- AdamW, learning rate `4e-4`, weight decay `1e-5`, clip `1.0`.
- Cosine schedule, 40 epochs, unchanged row-order fingerprint.
- Official validation, test-dev and test-challenge remain unread.
- The accepted frozen K1-v4 payload is reused; it is not retrained.

## Staging and gates

1. A Kaggle1 CPU job derives and atomically shards MoSE counts from the
   immutable fixed graph cache.
2. Local no-model acceptance verifies every part hash, source-index alignment,
   feature width, nonnegative counts and sealed-role flags.
3. Only after cache acceptance and after Kaggle1 has no other GPU training job
   may one isolated seed-42 candidate run. Accelerator identity is provenance,
   not a scientific-contract field; the actual T4 runtime must issue an
   accepted deterministic calibration certificate before training.
4. Mechanical acceptance recomputes the development MAE and paired delta with
   the immutable K1-v4 payload without model inference.
5. Promotion requires at least `0.003 eV` gain and a favorable paired interval.
   A pass grants only 500K-shortlist status; it does not authorize another seed,
   protected role, full training or submission.

## Cheapest falsifier

If cache generation is not tractable within one Kaggle CPU allocation, or if
the 100K gain is below `0.003 eV`, close this route.  Do not rescue it with
RWSE concatenation, motif subsets, hyperparameter tuning or extra seeds.
