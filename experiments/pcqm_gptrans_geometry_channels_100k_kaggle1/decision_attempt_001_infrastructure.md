# Attempt 001 infrastructure disposition

Kaggle1 kernel `nothingnessvoid/molgap-geometry-channels-100k-s42-v1`
(ID 136150940) ended `KernelWorkerStatus.ERROR` during the paired GPU
preflight. Both workers rejected the frozen full geometry initial-state
identity before any contracted training. The retained minimal remote log has
SHA256 `3a389bd6192f4d4653645d92036fc464499bd7c1cf220fbf3787b8023d62aa6b`.

The source package pinned the core checkpoint but reconstructed the geometry
basis buffers under the remote PyTorch runtime. Their full-state hash differed
from the desktop-frozen value. This is a source/runtime portability failure,
not evidence about bond distance or angle quality. Neither arm has a training
trace, selected checkpoint, development prediction, MAE, or replay entry.
Exact native device time was not reported. No official validation or test role
was read. The v1 source, Spec, receipt, and remote identity remain immutable.

The user authorized one repaired submission. Attempt 002 freezes and loads the
complete geometry initial state as a separate SHA-pinned artifact, preserving
the scientific recipe and pair comparison.
