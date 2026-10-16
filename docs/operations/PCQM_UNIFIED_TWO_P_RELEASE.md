# Unified V2: exact two-P compatibility and CPU role acceptance

On 2026-10-16 JST the human separately authorized only source structures2539039
and2980839 for evidence-qualified P compatibility and subsequent CPU data-role
acceptance. The original [Si acceptance](PCQM_UNIFIED_IDENTITY_REPAIR_ACCEPTANCE.md)
remains historical evidence, not silently reinterpreted. Live state is only in
`CURRENT_STATE.md`; platform evidence is separate under `two_p_release/`.
The [exact CPU launch receipt](../../platforms/_records/ims/pcqm_unified_roles_v2/two_p_release/cpu_launch.json)
pins actual submitted job, source, rule, authority and resource ceiling.

## Delta and reuse

Reuse `pcqm_molecular_identity` and `pcqm_unified_roles`, the immutable source
and graph bytes, 50K SQLite transactions, existing worker pool, hash ranking,
graph filtering, registered IMS runtime, atomic receipts and independent
acceptance. No trainer/optimizer/sampler/checkpoint/framework replacement.

The v4 identity rule is
`rdkit-canonical-isomeric-smiles-clear-atom-maps-known-ogb-si-two-p-explicit-h-v4`.
Only the two graph-proven P structures and canonical aliases join the31 proven
Si structures; ordinary sanitized identities stay unchanged. The ledger pins
the retained native explicit-H graph proof and a separately hash-bound direct
human-authority record. Unknown P structures/other error classes stay blocked.
Si-only rules cannot consume a P ledger, even with caller flags. No charge,
bond, hydrogen count, member or target is repaired/replaced.

Preparation uses a fresh v4 source/rule/output; no v2/v3 identity SQLite is
reinterpreted under changed keys/authority. Nested100K/500K/1M prefixes, hash
ranking, consumed-role exclusions,50K Common-Dev/50K Audit counts and protected
role guards stay fixed. Acceptance additionally verifies that Full is the
exact complement of held molecule groups and that held-out IDs follow the
frozen hash ranking, not just disjointness/counts.

## Boundary and gates

Run existing `unified_roles_cpu.pbs` with explicit immutable staging,
`MOLGAP_FALLBACK_LEDGER` and `MOLGAP_IDENTITY_ONLY=0`; use only CPU. Local tests,
complete source/ledger/authority pinning, native preflight and no duplicate job
precede submission. Every source transaction and derived graph chunk remains
independently recoverable. Complete accepted role/content evidence is required
for DATA_ROLE_ACCEPTED, not scheduler success or molecule parse success.

Private IMS preparation contains independently separated Audit payloads. No
official validation/test access, Audit scoring, model implementation, training,
GPU work, production/legacy conclusion change or platform mirror publication
is released by this repair. Missing native allocation cost stays explicit.
Stop on data acceptance or an actionable blocked condition; scientific-model
and indexed-loader release remain separate tasks.
