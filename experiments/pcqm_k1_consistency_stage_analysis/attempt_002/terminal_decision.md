# Terminal decision - 2026-10-08

NO_TRAIN. Corrected saved-artifact analysis accepted, not terminal training
acceptance. Original consistency500K pair remains ACTIVE_PARTIAL_STAGE.
No inference, training, remote action, model adoption or early-stop release.

Both immutable traces match23 contiguous epochs,89,838 optimizer steps,
11,499,264 sample presentations, LR/source/common Spec and all frozen shared
recipe fields. Per-arm source_config_identity differs legitimately with the
coefficient recipe. Objective arithmetic and exact stage-manifest hashes pass.

Consistency is better at10/23 same-step observations. Same-step differences
(consistency-control) are +5.519meV at1, -3.763 at13, -0.033 at21,
+2.656 at22, +1.668 at23. Independently selected stage checkpoints differ by
+1.141meV, not the last-step difference. Descriptive5-epoch blocks show early
loss, mid-stage gain and a later reversal; epochs are not independent repeats.

At23, supervised normalized L1 is slightly lower for consistency, and
disagreement0.003207 versus control0.003369 is4.79% lower. Weighted penalty is
0.3861% of supervised L1. Thus the term executes and reduces its observed
training disagreement; this does not prove clean endpoint generalization,
gradient insignificance, or a harmful loss at the final60-epoch endpoint.

Epoch windows total31,268.654s for coefficient0 and31,069.024s for0.1,
approximately0.64% difference. Both pay two-forward work; this is not a
single-pass speed test. Accepted T4 allocated ledger remains separately bound;
queue, external billing, CPU-core cost and phase/operator attribution stay unknown.

Worker wall2.723901s/process CPU2.656250s; accelerator/queue N/A for this local
arithmetic. Prior failed guard/source/cost unknowns are retained at the parent.
No row labels or prediction tensors were opened by this trace worker.

Next discriminator stays the original60-epoch selected raw/clean-BN pair.
The prefix cannot identify weight versus BN-state effects or calibrate stopping.
Accepted reusable arithmetic/evidence follows desktop non-promotion integration.
