# Full GPTrans-T Convergence Continuation

This experiment continues the accepted full-train GPTrans-T checkpoint instead
of restarting training. The immutable source is the 156,250-step, 20,000,000-
presentation run whose official-valid Gap MAE is 0.1090123132 eV. The source
checkpoint, model bundle and accepted official-valid result are bound by SHA256.

The continuation restores the raw model, EMA, AdamW moments, RNG state and
global row cursor. Only the learning-rate schedule is restarted: cosine decay
from 1e-4 to 1e-6 over at most six additional full-data-equivalent passes
(20,271,360 presentations). Physical batch remains 128, FP32 is retained and
TF32 remains disabled.

Official validation is explicitly consumed for convergence selection. The EMA
is evaluated every 26,395 optimizer steps. The run stops after three consecutive
non-improving evaluations or six evaluations, whichever occurs first. Test-dev
and challenge-test remain sealed. Consequently the selected validation result
is a convergence diagnostic, not an untouched leaderboard estimate.

The source run is never overwritten. Preflight, last checkpoint, per-pass
predictions, best bundle, trace, completion manifest and independent no-
inference acceptance are written under a new recovery root.
