# Bounded G1 scale qualification submission

On 2026-10-04 the shared source/recipe/initialization/upload release check passed
for local package `platforms/_records/kaggle/packages/gptrans_g1_scale_profile_v2`.
The Kaggle2 API returned [v1 physical receipt](submission_v1.json) with no invalid
source mounts or reconciliation requirement. The exact remote entry matched the
frozen local entry after SDK-local line-ending normalization; raw hashes and the
observed RUNNING status are in [remote verification](remote_kernel_verification.json).

This was a disposable execution-only profile under the [protocol](protocol.md),
not a 500K scientific training release. One visible worker observes two EMA
states on the same live model. All allocated devices count toward cost.
The failed local source-import check was retained in
[preparation failure](preparation_failed_binding_v1.json); no failed package was
submitted. Recovery changed only the shared SHA helper dependency, not data,
model, optimizer or profiling scope. The prospective plan bytes were preserved.

[Source publication](source_publication_v1.json), [release binding](release_binding.json),
[prospective trajectory](rml_plan/trajectory.json) and [exact monitor binding](monitor_binding.json)
retain the chain. The metadata-only acceptance wrapper is `accept.py`; it uses
the shared NO_TRAIN terminal transaction, records train-only role access and
native cost, and does not manufacture a training trace or causal MAE claim.
No automatic scientific successor was authorized.
