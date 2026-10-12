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

The actual submission receipt and monitor binding belong under `gpu/`. Live
job state routes through root `CURRENT_STATE.md`.
