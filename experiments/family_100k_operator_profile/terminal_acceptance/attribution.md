# Execution attribution - 2026-10-14

Accepted question: identify operator/cost hotspots for the selected existing
K1 and GPTrans shared-session recipes, not why100K gains disappear at500K.
Evidence owner: [decision](decision.md), [summary](summary.json), retained
profiles, canonical traces, native proofs and hash-bound mechanical reports.

- Data starvation is not supported in the sampled early window: loader waits
  were below0.2% of observed wall. No conclusion about every epoch is implied.
- Forward/backward and fine-grained CUDA dispatch dominate the sampled host
  timeline. Kernel fusion or reducing demonstrably redundant operations is a
  justified investigation, not an already verified semantics-preserving fix.
- Checkpoint construction/publication is visible but sampled once inside only
  16 updates. Its amortized full-run cost and asynchronous safety remain untested.
- GPTrans shortest-path preparation, clipping and optimizer work are visible;
  removing or changing them is not authorized by their timing alone.
- No profiler-off matched control or evaluation capture exists. Expected speed
  gains, GPU utilization and the cause of historical MAE drift remain
  `insufficient_evidence`; do not infer under/overfitting or transfer failure.

Cheapest next discriminator, requiring separate execution authority: a short
profiler-off original/proposed-execution pair on the same fixed batches,
initial tensors and runtime, first checking exact update/resume equivalence,
then timing steady-state and checkpoint intervals separately. Do not rerun a
full scientific contract just to finish this timing attribution.
