# Additive reference qualification

On 2026-10-01, the accepted observation reference was qualified using retained
metadata only. The original bundle and complete RML finalization were preserved
byte-for-byte. No model, checkpoint tensor, graph or protected role was loaded.

The [qualified bundle](reference_bundle.json) has a separate ID and distinct
role, cost and acceptance pointers. Role and cost sidecars contain exactly the
events in the original hash-verified terminal input and observed metadata;
missing CPU/queue measurements remain missing. Its scientific identity,
prediction tensor hashes, checkpoint, runtime, target transform and contract
are unchanged. The existing saved-prediction acceptance remains authoritative.

The [enrollment](../../rml_plan/reference_qualification.json) binds both original
and qualified artifacts. RML validates it against the immutable finalization
receipt and exposes the reference trace with its own accepted evidence ID,
rather than the historical V4 comparator ID. All observations, checkpoint
events, metrics and exposure counters are unchanged. Completion is verified
against accepted terminal counters, not by inventing a terminal trace event.

This qualifies a control, not a candidate winner. A reference without an
eligible canonical companion is still excluded from a replay *pair*. G1/G2's
original prospective bundle IDs and finalized paired-endpoint outcomes were
not rewritten; their replay admission remains pending. A new prospective
comparison may bind this qualified bundle after its ordinary release checks.
This record grants no training, extra seed, scale bridge or desktop handoff.

Reproduce the metadata-only qualification with the project Python and
`experiments/pcqm_gptrans_v5_audit_reference/qualify_reference.py`; repeat calls
verify the existing sidecars rather than replacing them. Run the focused
`tests/test_reference_qualification.py`, RML validation, rebuild and frozen check.
The retained-artifact tests require the original local reference evidence.
