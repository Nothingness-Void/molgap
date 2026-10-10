# K1 execution efficiency

Entry: [protocol](protocol.md), [evidence review](evidence_review.md),
[decision](decision.md). Reuses the K1 factory, normalized-L1 step, FP32/RNG
helpers, retained training-only probe and RML planner. No new trainer or platform
launcher. The active Kaggle payload is immutable and outside this question.

Reusable probes: `src/molgap/k1_execution_efficiency.py`.
Opt-in CPU layout: `src/molgap/k1_execution_layout.py`; defaults unchanged.
No production promotion, remote submission or development evaluation here.

Bounded continuation: [projection decision](projection_attempt_001/terminal_decision.md)
rejects project-before-gather/pack as direct strict-FP32 replacements. See its
matched numerical/cost evidence rather than extrapolating from algebraic equality.

[Isolated CUDA trace](kernel_attempt_001/terminal_decision.md) provides actual
SIMT-to-TensorOp kernel dispatch attribution; it does not qualify retrained MAE
or native accelerator throughput. Precision claims remain scoped to each attempt.

[CPU checkpoint-copy decision](checkpoint_io_attempt_001/terminal_decision.md)
qualifies an opt-in scalar-copy helper on retained state and synthetic IO fixtures,
not native sender throughput or end-to-end training savings.

[Single loss-scalar proposal](sender_scalar_reuse.md) removes a duplicate host
read in the frozen sender's next package, with CPU synthetic guard/counter parity;
it is not applied or GPU speed-qualified.

[Joint fused/layout/TF32 decision](combined_attempt_001/terminal_decision.md)
measures combination effects directly. It nominates a separately qualified
optimizer contract, not a default change or a retrained MAE result.

[Remaining host overhead](HOST_OVERHEAD.md) separates cached sampler metadata
from unqualified loader/RNG/history-publication costs; frozen senders stay unchanged.

[JSON publication sizing](JSON_PUBLICATION.md) isolates long-history encoding
cost and qualifies optional compact output without dropping fields or checkpoints.

[CPU loader attribution](loader_attempt_003/terminal_decision.md) separates
Windows worker startup from actual collation with exact TRAIN batch hashes.
It does not establish native GPU throughput or justify default worker changes.

[Persistent loader qualification](loader_reuse_attempt_001/terminal_decision.md)
measures cold versus retained workers with original epoch/resume indices. Reuse
is optional and limited to immutable deterministic datasets; native sender
training/resume qualification remains separate.
See [focused regression and bookkeeping limits](LOADER_REUSE_VERIFICATION.md).

[Packed-input qualification](loader_packed_attempt_001/terminal_decision.md)
extends the probe to the actual ConcatDataset shape and catches cold first-fetch
cost. The registered100K global-RNG loader is not silently switched to reuse.
See [packed regression and packaging verification](PACKED_LOADER_VERIFICATION.md).

[Global loader RNG compatibility](GLOBAL_LOADER_RNG.md) adds an explicit optional
Torch2.7.1 path with synthetic main-RNG parity, not registered-trainer adoption.
