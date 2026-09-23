# PairToken frozen-checkpoint scale attribution

Decision date: 2026-09-23. SCNet job `122843454` completed in 8m16s with
exit code `0:0`. Ten atomic 5,000-row chunks, the completion manifest,
source/checkpoint/cache hashes, row identities, targets, and role flags passed
independent no-model-inference acceptance. The 500K development baseline
reproduced the accepted PairToken prediction payload to a maximum absolute
difference of `9.54e-7 eV`. The compact terminal source is
`round3_terminal.json`; detailed per-row chunks and logs remain in ignored
`platforms/_records/scnet/k1_pair_token_scale_audit_122843454/`.

The frozen PairToken network relies on its learned relation update: suppressing
it raised MAE from `0.104304` to `0.250722 eV`. Uniform pair weighting raised
MAE to `0.125167`; diagonal-only pairs to `0.153419`; removing pair
normalization to `0.177894`. Off-diagonal-only pairs were close, but still
worse at `0.104821 eV` (paired difference `+0.000517 eV`). These are
out-of-distribution interventions on one trained network. They establish
network dependence on the mechanism, not that the mechanism improves over a
separately trained K1 model.

The exact matched K1 comparison from Round 2 remains unfavorable:
`0.103605` versus PairToken `0.104304 eV`, a PairToken regression of
`0.000699 eV` with a paired bootstrap interval entirely above zero. At the
shared epoch 48, PairToken had *lower training* MAE by `0.001611 eV` but
*higher development* MAE by `0.000699 eV`, making its train–development gap
`0.002310 eV` larger. An earlier epoch 20 still had a PairToken development
gain of only `0.001407 eV`, below this experiment's predeclared material gate.
The evidence supports a scale-specific generalization deficit rather than a
missing or unused relation branch. It does not identify a unique cause such as
capacity, regularization, or data chemistry without another controlled study.

The PairToken-to-500K promotion is closed. Do not extrapolate this 100K winner
to full scale, tune an inference-time gate on the reused development labels,
or submit a seed/architecture/optimizer variant under this protocol. K1 itself
is not rejected: its paired advantage over Full-GPS persisted from 100K to
500K in a different accepted comparison. Full-scale attribution belongs to
the desktop-owned, matched K1/GPTrans study; no matched full result is
established by this diagnostic, and the official role remained unread.

Native device/wall time was `0.137778 h`, queue time `0.006389 h`; peak
allocated accelerator memory was `98,529,792` bytes. CPU-hours were not
assigned from the allocation alone. The associated RML trajectory records the
terminal result. A 2026-09-24 validation correction preserved the original
noncausal prelaunch unchanged: a delivery experiment without a comparator does
not claim matched candidate identity, so its missing strict-comparison fields
cannot block noncausal record validation. The trajectory's planned native-cost
reference now names the already-recorded Round-1 cost event rather than the
separate budget-snapshot path. Repository-wide RML validation passes after
these corrections, but this attribution still has `NO_COMPARISON`, no V5
terminal evidence envelope, and no replay-ready or strict-causal claim. The
server compute-release helper still requires an actual reference bundle.
