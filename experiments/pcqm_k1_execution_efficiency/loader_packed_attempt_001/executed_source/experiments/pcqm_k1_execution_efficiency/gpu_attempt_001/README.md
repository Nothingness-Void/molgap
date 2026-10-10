# RTX 5060 execution qualification

Authorized 2026-10-10 by the user's request to test, under idle-only GPU access.
This is a new prospective sub-attempt of the existing execution question, not a
CPU rerun, scientific training, remote submission or model promotion.

Reuse the original frozen training-only 4096-row sample and initial state,
the corrected `run.py`, and shared K1 execution/layout helpers. Keep FP32/no-TF32,
BS128, seed42, normalized L1 and the family step. Three equivalence updates and
four balanced timing repeats (3 warmup + 12 measured steps) per case. Check all
loss/gradient/model/BN/optimizer/RNG evidence, then atomic checkpoint continuation.
Foreach/fused remain optimizer-contract changes even if numerically close.

Worker ceiling 600s, cooperative between operations; no persistent idle monitor.
The existing wrapper requires two idle observations before GPU allocation.
No development or protected roles. Scratch weights are discarded. Timing is for
pre-collated batches, not a full500K loader, EMA, evaluation or remote publication.
Local torch/PyG/device differ from native T4/A100; no throughput extrapolation.

Entry: `python experiments/pcqm_k1_execution_efficiency/gpu_check.py prepare`,
then `execute`, then parent interpretation and `close`. Do not edit/repeat the
completed original inputs or this attempt after execution.
