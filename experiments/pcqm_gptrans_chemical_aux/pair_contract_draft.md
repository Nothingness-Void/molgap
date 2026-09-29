# Conditional paired validation design

Status: DESIGN_ONLY, NOT_SUBMISSION_READY. No training is released by this file.

Two independent arms in one T4x2 allocation: the unchanged GPTrans-T control
and clean-input GPTrans-T with graph-level descriptor/fingerprint supervision.
The control is necessary for the requested same-allocation throughput and
identical-initialization causal comparison. The historical reference remains
contextual; do not use it to conceal the cost of the new control.

Freeze V4 100K/50K rows, source-index order, seed42, FP32/no TF32, BS128,
60 epochs, optimizer/scheduler, normalized Gap L1 and EMA development selection
from pcqm_gptrans_t_100k_v4. No corruption, PairNorm, geometry, readout change,
warm start or teacher. Initialize the shared backbone identically and construct
the auxiliary head in an isolated RNG context; preserve dropout/order RNG.

Proposed interface: read the existing 288-dimensional input of GPTrans readout
(256 node + 32 pair channels) during training only. One 288->32->712 head,
GELU between linear layers, gives 32,744 extra parameters (0.624% of 5,246,817).
Do not count a 256-dimensional hypothetical embedding as the raw graph state.
The interface should be an opt-in wrapper/hook, not a copy of GPTrans.forward.
Ordinary forward and exported state must use the original model without hooks.

Candidate loss proposed before performance measurements:
normalized Gap L1 + 0.1 * mean descriptor MSE + 0.1 * mean fingerprint BCE.
No paper-specific positive-class weights are copied from an unknown fitting
role. This unweighted BCE and compressed head are declared deviations from
the author implementation. A single design is tested, not a validation sweep.

Cache auxiliary labels only for accepted training source indices; join by
source_index, never batch position. Freeze names, versions, dtype, per-file
hashes, complete row coverage, failure ledger and input SMILES identity.
No PCQM data was read for the local feasibility checks. Actual cache access
must verify the frozen role and available assets first; no invented manifests.

GPU preflight after source and per-arm prospective binding: check zero-weight
auxiliary backbone update equivalence, real auxiliary gradients, finite losses,
state/resume and EMA restoration, exported Gap prediction equivalence and
absence of auxiliary keys/hooks. Compare synchronized optimizer-step time
and whole pipeline time, with warmup and repeated alternating measurements
on the same GPU. Final two-GPU packing must also verify resource contention.
Engineering target: <=5% training-time increase; no extra exported forward
operations and no statistically resolved inference slowdown. CPU cache time,
GPU wall time, memory and actual device assignment are recorded separately.

Required trainer gap: _optimizer_step currently hardcodes Gap L1. Extend a
small opt-in loss adapter while retaining the baseline path, Gap-only metric,
separate auxiliary metrics, clipping semantics, checkpoint/EMA state and source
identity. Do not monkeypatch the loss globally or mislabel total loss as Gap MAE.
The current static registry has no chemical-aux addon; registration, focused
tests, exact executable source, Spec v2 same_run_replay and both prospective
arm plans must precede GPU submission. A metadata adapter_probe is insufficient.

Accuracy gate: at least 3 meV paired selected-endpoint improvement with positive
paired uncertainty interval, plus the cost gate. Capture exact steps/sample
presentations and late-prefix development differences; do not stop using an
uncalibrated curve rule. Row bootstrap does not establish seed robustness.
Any 500K or extra seed requires a separate decision; no automatic promotion.
