# Infrastructure-only retry of the frozen motif K1 screen

Version 1 stopped before graph loading, model construction, preflight or any
scientific epoch. Its pinned software gate asserted the requested *P100* name,
but Kaggle reported only generic `Gpu` metadata and the runtime assertion
failed at approximately 174 seconds. The archived log does not identify the
actual assigned device. Therefore version 1 is an infrastructure failure,
not a negative model result, and no checkpoint can be resumed.

This version retains the exact [GPU scientific contract](../gpu_training_contract.json),
model, sidecar, seed, row order, optimizer, selection and reference. The only
release change is device qualification: request one P100 but allow either one
actual P100 or one actual T4 after the pinned software tuple succeeds. Record
the actual GPU before any model work. Every other device and every software
mismatch fail closed with a diagnostic output. Both allowed devices have 16GB;
the existing 15% peak-memory-reserve preflight still applies. No second arm is
invented just to occupy T4x2, and no precision/batch change is permitted.

Version 2 has a new immutable source commit, source package, physical run ID,
prospective RML trajectory and native cost event. Version 1 logs and receipt
remain untouched. There is no scientific-success claim until the version-2
training artifacts pass independent terminal acceptance.
