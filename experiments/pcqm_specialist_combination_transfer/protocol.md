# Frozen protocol: K1/PairToken combination transfer

## Question

Can the large label Oracle headroom in the molecular specialist audit be
converted into a usable prediction rule without another encoder run?

This is a bounded three-round, no-model-inference analysis.  The two experts
are immutable K1-v4 and original PairToken at 100K, and their separately
trained but exactly matched 60-epoch K1 and PairToken endpoints at 500K.
Only the official-train-derived development roles are used: source indices
`[100000,150000)` for rule fitting and `[500000,550000)` for transfer
evaluation.  The 500K role has already selected both model checkpoints;
therefore its result is development evidence, not an unbiased final estimate.

## Identity and alignment gate

- The accepted 100K aligned matrix must have SHA-256
  `c52e944def35d8d9ccb41011c9cf820182702c1a9006189b6fc488397169edbb`.
- The 500K K1 payload must have SHA-256
  `e43728a30a74a19e9e969cef3e94f3c0f9718a1c606bfa5306bd9b3c87ca8d31`.
- The 500K PairToken payload must have SHA-256
  `35fdaa76f10e166ccb123b6005929a420f18b2678a286023d4aa62e48ec9ca01`.
- Within each scale, source indices and target tensors must match exactly;
  500K indices must be the contiguous range `[500000,550000)`.
- Predictions and targets must be finite.  Any failure stops all rounds.

The 100K model pair was selected from the prior specialist audit.  The 500K
models have the matched60 scientific contract, but the 100K-to-500K change
also changes training membership and exposure.  No cross-scale strict causal
claim follows from this transfer test.

## Three fixed rounds

1. **R1 fixed equal:** use `0.5*K1 + 0.5*PairToken` at both scales.  Report
   MAE versus both single models and the 500K paired row-bootstrap interval.
2. **R2 one global weight:** on the 100K role, select the K1 weight from
   `{0, 0.025, ..., 1}` minimizing MAE, with ties going to the value closest
   to `0.5`.  Evaluate five source-index folds by refitting on the other four;
   then fit once on all 100K rows and transfer that weight unchanged to 500K.
3. **R3 disagreement bins:** on the 100K role, derive five quintile bins from
   `abs(K1 - PairToken)`.  In each bin choose a K1 weight from `{0, 0.5, 1}`
   minimizing bin MAE.  A bin may deviate from the R2 global weight only if
   it has at least 5,000 fitting rows and improves fitting MAE by at least
   `0.001 eV`; otherwise use the R2 weight.  Evaluate by five-fold refitting
   on 100K.  Fit once on all 100K and transport thresholds and weights
   unchanged to 500K.  Binning uses predictions only; labels are confined to
   fitting and scoring roles.  No descriptor, SMILES or ground-truth error
   may enter the decision at inference.

All three rules and gates are fixed before calculating 500K fusion metrics.
No method may be selected or tuned on 500K and rerun as if it were a holdout.
Report each round even if an earlier round is negative.

## Decision gate

Use the stronger single model at 500K as the shared benchmark.  A rule earns
only an exploratory continuation recommendation if its 500K gain is at least
`0.001 eV`, its paired 99% row-bootstrap interval for gain is wholly positive,
and gain is positive in all five source-index folds.  R2 or R3 also needs at
least `0.001 eV` extra 500K gain over the simpler preceding rule to justify
its added fitting complexity.  Bootstrap estimates row uncertainty only, not
training-run variance or checkpoint-selection uncertainty.

Failure closes learned molecular routing from this evidence.  Success is not
production promotion and does not open official validation, test-dev,
test-challenge or full training.  The result is a `NO_TRAIN` diagnostic
trajectory; the RML model-training replay pool is inapplicable to these three
post hoc prediction rules.
