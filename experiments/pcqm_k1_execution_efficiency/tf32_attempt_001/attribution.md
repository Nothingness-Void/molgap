# Precision and execution attribution

The aligned layout comparison isolates TF32 permission more cleanly than TF32
against baseline: context, transfer, optimizer and explicit-barrier behavior match.
Its four positive windows are a local short-run signal, not training stochasticity
or a validated long-run screening rule. TF32 affects matmuls only; unchanged
pointwise/indexing/normalization and optimizer execution limit total benefit.

Frozen predictions change by about0.000116eV on average even while the TRAIN
sample MAE change is almost zero. This constrains numerical sensitivity of one
accepted state, not training dynamics or generalization. Three scratch updates
already differ in gradients/state; no strict equivalence or quality adoption.

The K1 path repeatedly invokes small projections and unfused AdamW over hundreds
of parameter tensors. Instrumented phase timings show meaningful optimizer cost;
CPU traces contain many mm/addmm, fill, copy, index and pointwise calls. Missing
CUPTI activity prevents attribution to actual GPU kernels, occupancy or Tensor
Core utilization. CPU self-time must not be mislabeled GPU compute time.

Mechanical review found the old adapter could trust complete top-level status
despite failed cases. Actual retained cases all succeeded, so this run is not
missing comparisons. A separately tested closure guard now checks each case,
timing count and unchanged-FP32 exactness. The executed source is preserved;
the guard was not retrospectively claimed to have run inside the worker.

Next minimum discriminators: native profiler support to inspect dispatch;
prospectively bounded projection-before-gather/packing numerical falsifiers;
separately authorized matched TF32 training-quality comparison if required.
Do not lower exactness gates, retrain a reference or consume protected roles just
to fill the remaining gaps. Recommendation and active payload stay unchanged.
