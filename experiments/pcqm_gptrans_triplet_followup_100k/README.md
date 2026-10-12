# Triplet follow-up100K

The [frozen training protocol](protocol.md) owns the two hypotheses and their
unchanged aggregate-reference recipe. `freeze.py` and `accept.py` are thin calls
to the existing prospective, release, native acceptance and RML owners.

Before compute release on2026-10-13, the mandatory
[separate post100K action](gpu/post100k_audit_action.json) was frozen under
`SCREENING_POLICY.md`. It uses the full500K internal-development role only
after100K training acceptance. It supersedes the training protocol's statement
that this separate plan still needs to be supplied; it does not change either
training role or release500K optimization. Candidate checkpoint hashes cannot
exist prospectively: the deferred action requires their actual accepted values,
a linked NO_TRAIN plan and verified matching loader before its own release.
It is not claimed executable or terminal-complete in advance.

The [partial terminal interpretation](gpu/results/interpretation.md) owns the
saved-output comparison and learning-curve analysis. The complete drop arm's
`gpu/degree_local_triplet_drop_ema999/results/` retains its native acceptance,
strict comparison and actual Replay admission; the product arm's
`gpu/degree_local_triplet_product_ema999/failure_results/` retains its execution
failure, bounded diagnosis and allocated idle cost, without a training Replay
claim. Neither the failed mechanism nor the entire dual is relabeled accepted.

The actual submission receipt and monitor binding belong under `gpu/`. Live
job state routes through root `CURRENT_STATE.md`.
