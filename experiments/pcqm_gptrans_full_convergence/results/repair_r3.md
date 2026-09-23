# Continuation infrastructure repair

The GPTrans R2 failure occurred because loading an EMA checkpoint replaced GPU
buffers with CPU tensors. The loader now validates keys, shape and dtype and
copies the restored values onto each existing destination device.

The K1 R1 payload omitted pcqm_k1_full_runner.py. Packaging now includes the
complete tracked Python runtime, checks required entry dependencies, compiles
every source and emits per-file SHA256. The uploaded archive and extracted
files are verified before actual entrypoint import checks in the IMS environment.

Both GPU preflights restore the accepted source model and optimizer, execute
an update, save a disposable checkpoint, load it through CPU and execute another
update. GPTrans also restores and updates EMA twice. Training requires the new
roundtrip acceptance flag, so older incomplete preflights cannot authorize it.
These disposable steps do not advance the training checkpoint cursor.

Preflight and continuation execute sequentially in one GPU allocation. Frozen
source checkpoints, architecture, scientific contracts and stop rules remain
unchanged. Acceptance is a dependent CPU job. Failed attempt directories remain
intact. New roots are gptrans_convergence_20260915_r3 and
k1_convergence_20260915_r2 under the authorized IMS project root.

Local validation: 13 tests passed, including tensor-only EMA destination-device,
update, no-aliasing and invalid-shape regression checks. No local encoder ran.
Remote GPU roundtrip remains pending until scheduler execution.
