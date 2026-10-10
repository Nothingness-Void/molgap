# Projection placement falsifier

Desktop, execution-only continuation of the user-authorized K1 efficiency review.
Prior exact CPU-layout and TF32/optimizer diagnostics leave repeated projection
work as a specific untested mechanism. This is not a rerun of those completed
questions. Query prior RML/canonical attribution before changing projection order.

Mechanisms, separately applied: persistent edge source/target projections on real
nodes before endpoint gathering; bias-free mixer key/value projections on real
nodes before dense padding (using the qualified CPU-layout context). Same family
weights, architecture, dropout/BN, FP32/no-TF32, BS128, seed42, unfused AdamW and
normalized Gap L1. Algebraic equivalence is not numerical equivalence: require
exact three-update loss, pre/post-clip gradients, full model/optimizer and RNG
before nominating a same-contract path. A mismatch stays rejected for that gate.

Reuse the frozen initialization/4096 train-only probe and family step. Four
balanced timing repeats (3warmup+12measured steps), include context/layout and
transfer/guards/RNG costs. No loader/full-epoch extrapolation. CPU synthetic tests
cover hooks and shape/RNG restoration; no development/protected roles or learned
candidate fitting. Only first1920 members are used in scratch execution.

User permits short idle-only local GPU diagnostics; worker600s cooperative ceiling,
CPU4threads, no persistent monitor, remote submission or process termination.
No TF32/AMP or optimizer variation in this attempt. If GPU is occupied, do not
launch accelerator work. Record decoded/executed members and measured scope.
NO_TRAIN terminal; CPU layout is the matched comparator for project-before-pack.
`gather_control` is the original-layout comparator for project-before-gather,
with an identical explicit-sync context. No automatic quality test, model promotion or
native T4/A100 admission. Parent attribution precedes another unrelated module.
