# Sparse Atom--Bond Dual Stream Seed-42 Decision

## Decision

The candidate passed the complete no-inference artifact acceptance but failed
the scientific gate. Close this sparse atom--bond dual-stream mechanism. Do
not run seeds 43/44, a width/depth/LR retry, full-data training, official
validation/test-dev, or molecular-research-server work from this result.

## Evidence

The frozen distance-plus-angle comparator achieved `0.1353926807641983 eV`.
The dual-stream candidate achieved `0.13646799325942993 eV`, a regression of
`0.0010753124952316284 eV` (`0.7942%`). It used 5,083,889 parameters, 192,832
more than the comparator (`3.94%`), while mean throughput fell from `403.25`
to `357.55 graphs/s` (`-11.33%`).

The run completed all 40 epochs within the 14,400-second budget. Its best
epoch was 37 and the final epoch was worse than the best, so the failure is
not explained by early termination or an unspent training horizon. Preflight
proved exact shared-backbone initialization, zero new residual injections,
finite forward/loss/gradients, and the expected parameter count. Validation
row and target hashes matched the frozen comparator exactly.

## Interpretation

The extra bond stream did learn, but separately normalized sparse attention
over the same accepted non-backtracking wedges did not add sufficiently new
chemical information. The accepted backbone already updates persistent bonds
from atom states and persistent wedge states in every layer. Four additional
bond-attention/exchange blocks therefore mostly reprocessed information that
was already present, adding optimization freedom and sparse reductions rather
than a new invariant. The near-end best epoch followed by regression also
indicates ordinary variance/overfitting rather than a capacity shortage that
would justify a longer run.

Under the frozen literature ranking, the next distinct bounded question is a
deterministic ring/conjugation hierarchy. It must receive a separate protocol
and cache contract before submission; this decision does not authorize that
task automatically.

The post-hoc curve and checkpoint attribution is frozen in
[`../../post_dual_stream_failure_attribution.md`](../../post_dual_stream_failure_attribution.md).
It establishes that the new modules learned but widened the generalization gap,
and therefore closes optimizer, width, depth, head-count, and extra-seed retries.
