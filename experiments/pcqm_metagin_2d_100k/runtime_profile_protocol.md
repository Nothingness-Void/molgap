# 2026-09-28 bounded MetaGIN runtime diagnosis

The v1 training attempt stopped at a cold-step cost gate. One no-result GPU
profile may use the accepted 100K training role and the already accepted
topology sidecar to measure 8 warm-up and 72 steady-state full FP32/BS128
optimizer steps, followed by 32 forward-only training-role batches. It must
not run a full epoch, select a checkpoint, read official validation or
protected tests, or emit a development metric. Hard profile wall limit:
1,200 seconds after dataset setup. The fixed scientific training contract
remains untouched.

Decision: a six-hour GPU retry of the same 4 x 256 architecture is admissible
only if the observed 40-epoch projection with a 20% contingency is below
21,600 seconds and at least 15% VRAM remains. Actual hardware and all visible
allocated devices count toward the native cost. If that gate fails, this
architecture size is deferred; a smaller MetaGIN-family model would require
a separately frozen question and prospective source/trajectory identity.

The profile uses no model-selection outcome and cannot be compared with the
immutable K1 validation result. It repairs uncertainty about execution cost,
not scientific quality.
