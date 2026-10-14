# Terminal custody - 2026-10-14

Both original prospective trajectories are finalized through the existing
`close_terminal_multi_arm`/RML lifecycle. Immutable prospective, submission,
raw output and dated acceptance records remain unchanged. See
[K1 receipt](../rml/k1_mean2/rml_finalized/finalization.json) and
[GPTrans receipt](../rml/gptrans_local/rml_finalized/finalization.json).

Execution is COMPLETE and validity VALID; scientific disposition is
INCONCLUSIVE because this is an operator diagnostic without a frozen strict
reference. Retained canonical traces have40/60 observations and explicit
`reference_id=null`, `eligible=false`, `missing_frozen_reference`. No strict
cross-family comparison, model adoption, READY or replay-ready claim is made.

Observed train/internal-development role events are bound to retained runtime
evidence. Official validation/test roles are untouched. Pair entry allocation
is6.682364 T4-device hours, apportioned once across its two independently
assigned arms; CPU, queue and GPU busy time remain unknown.

Independent Luna ran RML `validate`, `rebuild` and `check --frozen` successfully:
105 evidence,117 trajectories,238 costs,295 roles and33 traces. The existing
unreferenced two-arm custody regression passed (1test). Portable Git checking
requires the new inputs/derived outputs to be committed; its subsequent result
is separate from these precommit checks.

Compact canonical evidence, profiles, hashes and acceptance metadata belong in
Git. PT payloads and large raw profiler traces remain ignored locally and
independently retrievable from the exact private Kaggle version1 outputs.
`metadata_retention.json`, per-arm retention receipts and output manifests pin
their bytes. Clone-only metadata does not conjure missing weights/raw traces;
retrieve/hash-check them before an artifact-level replay or native inspection.

Reusable profiler/local-bond wiring and accepted diagnostic evidence follow
the non-promotion desktop integration route. The additional
[topology reuse patch](execution_optimization.md) has separate verification;
this terminal closure does not certify a new GPU execution identity.
