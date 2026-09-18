# Protocol

## Question

Does parameter-free normalization of each GPTrans pair update retain the
positive 100K effect at the fixed 500K scale?

## Frozen comparison

- Candidate: GPTrans-T 12x256, pair width 32, `pair_update_norm`.
- Parameters: 5,246,817, identical to the accepted GPTrans reference.
- Roles: source rows `[0, 500000)` for training and `[500000, 550000)` for
  internal development.
- Seed 42, deterministic FP32, TF32 disabled, one device, physical BS128,
  drop-last, normalized Gap L1, AdamW and the accepted 60-epoch cosine
  trajectory.
- Model selection: lowest internal-development MAE observed before the run's
  terminal boundary.
- Sealed: official validation, test-dev, and test-challenge.

## Pre-registered futility rule

Sixty epochs is the ceiling for a competitive candidate, not mandatory work
for an obvious loser. Compare candidate best-so-far against the accepted
GPTrans reference best-so-far at matched prefixes:

| Completed epochs | Reference best | Stop when candidate deficit is at least |
|---:|---:|---:|
| 30 | 0.1125212312 eV | 0.006 eV |
| 40 | 0.1082282215 eV | 0.003 eV |

The reference trace SHA256 is
`22cb2bea6ee531402b951334fb791f091ec64c68b3e6f539fe1c9851dcdce1d4`.
Stopping produces `FUTILITY_STOPPED`, never `COMPLETE`, and cannot nominate a
model. A candidate that passes both gates continues on the unchanged schedule
to epoch 60. No plateau-based patience rule is used.

The rule was calibrated before launch. On the frozen 100K traces it keeps the
accepted `pair_update_norm` candidate and stops the rejected `pair_post_norm`
at epoch 30. Historical accepted 500K GPTrans, EdgeState, and K1 traces show
late improvements, so ordinary patience early stopping is unsafe.

## Decision gate

- `FUTILITY_STOPPED`: reject at 500K; no continuation or full-scale release.
- `COMPLETE`: perform mechanical acceptance and paired row-level comparison
  against the frozen GPTrans 500K predictions.
- Nomination still requires at least 0.003 eV final gain. Passing a futility
  gate is not evidence of improvement.

