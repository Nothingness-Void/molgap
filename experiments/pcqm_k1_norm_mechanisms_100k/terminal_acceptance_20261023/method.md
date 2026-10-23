# Acceptance execution, 2026-10-23

Reused `retrieve_exact_metadata` and `retrieve_pinned_exact_files` for exact
Kaggle version3 allowlists; no output listing, bulk output, cache, historical
chunk or scratch model collection was downloaded. The new scope/gain retrieval
is142.80MiB onD:, principally the two selected models, aligned predictions and
full final resume checkpoints. Existing position outputs were reused.

Reused `TargetIdentityBinding.from_frozen_package` and `accept_workflow` /
`inspect_session_output` for CPU-only mechanical acceptance. This loads retained
tensors safely, verifies finite outputs and hashes, and never constructs a
model or executes inference. The v3 receipt's original pretty serialization
was preserved; a canonical serialization of the identical object was supplied
to the receipt reader. Transport receives integer3; RunContext retains its
schema's string version"3". Neither conversion changes job identity.

The local checks verified runtime certificate/provenance fingerprints, source
package identity and separately attested three-file repair; scope actual-resume
certificates bind the original checkpoint hashes and exact retained trace
prefixes. Native scratch tensor proof payloads remain remote and were not
independently revalidated locally. That limit is retained in terminal evidence.

Saved-prediction comparisons reuse `_compare_paired_errors` with2000 row
bootstrap samples, seed42, and `_optimal_constant_shift` for the disclosed
post-hoc diagnostic. Canonical traces and scheduled norm diagnostics provide
matched endpoints. No training, inference, protected-role access, new unit test
or regression-suite run occurred during acceptance.

The generic Spec terminal descriptor/stager assumes an observed attempt was
already listed in the original prospective action. This continued job has a
separately authorized v3 receipt, while the original action lists v1. Reviewed
existing-schema terminal packages preserve that actual v3 observation and use
the existing `close_terminal_multi_arm` / `finalize` owners. Original prospective
bytes and source identities remain preserved. No validator was relaxed and no
new RML schema or finalization framework was introduced.

Physical entry windows are the cost authority. Versions1,2,3 are counted once,
including the failed v2 preflight; nested partial-ledger ancestry is not added
again. The per-arm event records its dedicated device share and leaves unknown
CPU, queue and per-arm wall measurements missing.
