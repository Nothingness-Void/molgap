# K1 relation-resolution study

## Question and frozen scope

This September 27, 2026 server study tests three pure-2D architectures in two
Kaggle2 notebooks, not three optimizer searches. It reuses the immutable K1
reference; no baseline training, geometry, teacher, pretraining, or full-scale
release is authorized. One seed (42), physical BS128, FP32/no-TF32, 40 epochs,
31,240 optimizer steps and 3,998,720 sample presentations apply to every arm.
The exact executable identities live in `training_contract.json` and each
arm's prospective RML plan. These reused internal development roles are not
sealed scientific holdouts.

## Evidence and falsifiable interventions

The original layer-6 PairToken retained nonlinear pair features but pooled a
single molecular token before returning it to all atoms. Node-conditioned
broadcast cannot recover pair identity after this pooling. The closed
node-query linear-attention experiment changed all three global exchanges
and lost its early advantage at terminal exposure and on frozen500K dev.
Thus its failure does not test receiver-specific nonlinear pair messages at
one layer. Authorities:

- `../pcqm_k1_pair_token_100k/decision.md`
- `../pcqm_k1_linear_attention_100k/STATUS.md`
- `../pcqm_k1_spd_pair_token_100k/decision.md`
- `../pcqm_k1_sparse_triplet_100k/decision.md`
- `../pcqm_k1_portability_dual_100k/STATUS.md`

The three arms preserve all original K1 local/slot layers and add one
zero-return-initialized residual at layer 6:

1. **Receiver-pair**: normalize nonlinear ordered-pair weights separately
   for each receiving atom, then deliver that atom's own pair-weighted vector.
   This is not a new graph gate or another common token.
2. **RRWP-pair**: the receiver-pair arm plus one learned encoding of
   `[I,R,...,R^7]` from the existing real-bond adjacency. R is row normalized;
   isolated atoms stay at themselves. This is an internal encoder operation,
   not a new stored input feature or cache. Raw atom/bond/RWSE16 fields,
   accepted shards, target transform and row order remain byte-identical.
   Unlike SPD, RRWP preserves multiple-walk relative connectivity.
3. **Triplet-aggregate**: the receiver-pair arm plus one inward vector
   aggregation `o_ij = sum_k softmax_k(b_ik) sigmoid(g_ik) V(p_jk)`.
   It tests pair-to-pair information flow, not the rejected all-depth wedge
   memory or low-rank Hadamard triangular product. It does not recover true
   3D bond angles from 2D labels.

Primary algorithm sources are [GRIT](https://github.com/LiamMa/GRIT) and
[TGT section 3.1](https://arxiv.org/html/2402.04538). These are deliberately
bounded adaptations, not reproductions of their entire models or leaderboard
scores. Shared code/initialization between arms allows RRWP-vs-receiver and
triplet-vs-receiver paired endpoint attribution, in addition to frozen K1.

## Release, measurement, and stop rules

- Validate the real reference bundle and every evidence pointer before release.
- Pin the accepted Torch/PyG runtime; qualify each assigned GPU at runtime.
- Fail closed on cache SHA, row order, target transform, parameter identity,
  initialization parity, masking, equations, gradient or resume-equivalence
  checks. Require at least 15% free device memory in optimizer preflight.
- The dual notebook uses two T4s, independent child processes, RNGs,
  optimizers, directories and device visibility. The RRWP notebook uses one
  GPU; a fourth arm is not justified just to consume the remaining device.
- Preserve checkpoints each epoch, native canonical RML observations and
  role truth each epoch, independent recovery archives every ten epochs,
  observed worker cost and failures. Runtime errors are infrastructure
  outcomes, not scientific negatives. No automatic blind retries.
- Hard safety budget: six wall hours per training worker. Estimated training
  cost is 2–4 device-hours per arm, unverified until remote timing; maximum
  training allocation across the three workers is eighteen device-hours,
  plus installation/qualification overhead. A separate audit is capped at
  1.5 device-hours. No paid excess or account switching is authorized.
- Report terminal and matched-exposure curves, not only the best early epoch.
  The predeclared conservative material screen gate remains 0.003 eV vs K1
  for this study, not a universal V5 constant or measured noise estimate.
  Smaller positives stay `POSITIVE_BELOW_GATE`; do not lower the gate later.

## Separate post100K portability audit

Training kernels stop after publishing training artifacts. The controller
must perform no-inference acceptance first. Only accepted immutable best
checkpoints may enter the separately planned `NO_TRAIN` audit. Reproduce the
original dev predictions, then evaluate the full fixed500K internal dev
50,000 rows `[500000,550000)`, using the accepted reference prediction if
available. No optimizer, training-label scan, random resampling, official
validation, test-dev, test-challenge, or sealed role is permitted.

Use the same model/software/normalization; do not label this 500K *training*
or claim scale robustness from it. Report gain sign and retention; 500K/full
training and seeds43/44 require a new explicit decision. Every arm receives
its own prospective→trace→evidence→cost→role→terminal chain. Replay-ready is
claimed only after shared terminal validation and replay-pool rebuild pass.
