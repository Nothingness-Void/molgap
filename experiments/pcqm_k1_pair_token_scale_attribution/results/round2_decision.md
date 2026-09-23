# Round-2 aligned scale comparison

Decision date: 2026-09-23. This is a saved-prediction analysis; no model was
trained or inferred. Source: `round2_residual_analysis.json` and the accepted
K1 reference job `122743291` versus accepted PairToken job `122312462`.

The 50,000 internal-development source indices and targets match exactly on
the fixed 500K manifest SHA-256
`630d30046d6cdc1f91fb169cd1eb4720bd5b352dc1ebeb641a11ead1ebae9751`.
Under the same seed-42, FP32, BS128, matched60-v4 contract, K1 has Gap MAE
`0.1036051948 eV` and PairToken has `0.1043037290 eV`. PairToken is worse by
`0.0006985342 eV`; a 10,000-resample paired row bootstrap gives a
candidate-minus-reference 95% interval of `[0.0001388103, 0.0012531244] eV`.
Its row win rate is 49.11%. The earlier 100K paired gain was
`0.0030435771 eV`; therefore its positive gain did not transfer to this 500K
contract. The old positive 500K comparison used an unmatched scalar K1
reference and is not the causal scale comparison.

Error-quantile analysis suggests PairToken helps K1-hard rows but harms
K1-easy rows; those bins are defined using observed K1 label errors, so they
are diagnostic only and cannot be used as an inference-time router. The fixed
Round-3 gate is met. Exactly one frozen-checkpoint inference intervention is
released to ask whether the learned relation token remains active; no new
architecture, retraining, seed expansion, full run, or protected-role read is
released. Native CPU time for this local saved-prediction analysis was not
measured; it must not be backfilled from wall-clock inference.
