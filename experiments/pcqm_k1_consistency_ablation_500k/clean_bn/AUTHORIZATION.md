# Selected-state clean-BN diagnostic

2026-10-09. User authorized completion of the predeclared secondary diagnostic
in the [original protocol](../protocol.md). This is the same question, not a
new training run, baseline retraining, epoch reselection or promotion.

## Frozen procedure

Both selected epoch49 states (zero-based48) receive exactly the same procedure:
16384 training members, numpy.default_rng(20261008), uniform without replacement
in draw order, physical batch128, FP32, no TF32, CPU with four intra-op threads.
Only BN running buffers adapt cumulatively with dropout disabled. Learned
parameters remain frozen; no optimizer, gradients or label objective.
Full50000 consumed internal-development members are predicted before, during
and after calibration. Saved native predictions must reconstruct within1e-4eV;
the restored full prediction and original buffers must match exactly.

Ceiling:600seconds per arm,1200seconds per pair, sequential execution. On failure,
retain partial evidence and stop; no automatic retry. CPU process time and wall
time are measured separately; GPU/device and queue time are not applicable.
The bound input JSON is published before model/data execution. Original
trajectories, selected checkpoints, native predictions, source inventory,
protocol, diagnostic code and the graph manifest are pinned by SHA256.

Only the existing train prefix0:500000 and already-consumed internal development
500000:550000 are authorized. Packed training labels are decoded by the existing
reader but not used in BN calibration. Development labels support diagnostics;
this is not independent validation. Official validation and all protected roles
remain untouched. CPU software differs from native T4 software and is reported.

## Interpretation and qualification

Primary raw endpoint retains the original1meV material gate and paired-row
interval. Secondary results compare both calibrated arms, and each arm before
versus after. Never substitute a one-arm calibration gain for a consistency-loss
gain, or interpret row bootstrap as training-seed variance.

The original two prospective arms declare no frozen reference identity and both
are labelled candidate. Preserve that defect and original source bytes.
Scientific/mechanical acceptance does not establish strict V5 training replay.
Canonical trace finalization must not omit retained training traces or invent a
reference to work around schema/validator requirements.
