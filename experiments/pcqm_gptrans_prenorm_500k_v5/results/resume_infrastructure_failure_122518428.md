# Reference continuation infrastructure failure

Decision date: 2026-09-18

Reference continuation job `122518428` failed before loading the epoch-37
checkpoint or executing resumed training. Its redundant fresh calibration
observed zero loss delta and a one-ULP parameter delta of
`1.1920928955078125e-7`, narrowly above the frozen `1e-7` threshold. The
failure produced no scientific result and did not invalidate the accepted
runtime certificate or the complete epoch-36 checkpoint.

The repair does not loosen the threshold or retry the stochastic gate. V5
allows one accepted runtime certificate to be reused for the same
platform/software/runtime tuple. The certified-resume path therefore verifies
the checkpoint's scientific contract, source/archive identities, runtime
certificate ID and current runtime fingerprint, then restores model, EMA,
optimizer, RNG and epoch cursor exactly. The current execution commit/archive
are recorded separately from the frozen scientific source.

`src/molgap/gptrans.py` and `src/molgap/gptrans_variants.py` are byte-identical
to frozen scientific commit `dff104723222346e703cd98ef71c19884f46076c`.
This is an infrastructure-only continuation repair, not a model, data,
optimizer, schedule, precision, batch, tolerance or role-access change.
