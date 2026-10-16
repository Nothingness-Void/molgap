# Unified V2 molecular identity compatibility

## Scope and reuse

On 2026-10-16 JST the user authorized repairing the Si canonicalization blocker,
preserving the database/training members and original failure evidence. CPU
identity preparation/acceptance only; no model implementation, scoring, GPU,
training, official validation/test access, or automatic scientific successor.

The fixed-data owner reuses its source SHA checks, 50K SQLite transactions,
worker mapping, member selection, graph filtering, acceptance and atomic IO.
The new identity adapter uses the already accepted scale-cache strategy
(`sanitize=False`, `UpdatePropertyCache(strict=False)`), not its graph-builder
function: that function returns OGB features rather than a canonical molecule.
It does not modify the historical cache implementation, trainer or optimizer.

The evidence is the existing
[ten-row audit](../../experiments/pcqm_k1_scale500k/parser_fallback_audit_v4.json)
and [accepted cache receipt](../../experiments/pcqm_k1_scale500k/terminal_handoff_marker_cache_v5.json).
Strict Si parsing is not globally unsupported: ordinary tetravalent Si works.
The old strict-v1 canonicalization omitted compatibility for known exceptions.
RDKit's [valence rules](https://www.rdkit.org/docs/RDKit_Book.html#valence-calculation-and-allowed-valences)
and [sanitization caution](https://github.com/rdkit/rdkit/wiki/FrequentlyAskedQuestions#explicit-valence-for-atom--is-greater-than-permitted)
explain why parsing success alone does not establish chemical correctness.

## Frozen compatible identity rule

`rdkit-canonical-isomeric-smiles-clear-atom-maps-known-ogb-si-v2` adds explicit
`fallback_ledger_sha256` and `molecule_parser_sha256` pins to the existing rule.
Pin actual archived bytes, RDKit runtime and source commit before any selection.
Use a new output root; never resume the failed strict-r2 SQLite inventory under
changed semantics. The historical rule remains supported, not reinterpreted.

Normal sanitized canonical keys are byte-identical to strict v1. Fallback is
limited to structures whose only reported chemistry problems are Si explicit
valence exceptions AND whose canonical isomeric key belongs to the frozen,
previously accepted ten-row ledger. Equivalent atom ordering/map-number aliases
are covered regardless of source index; otherwise duplicates in another role
could escape detection. Do not append a parser-mode prefix to molecule keys.

No bonds, charges, hydrogens, isotopes or stereochemistry are repaired/guessed.
Fallback round-trip and reversed atom ordering must agree. Unknown structures,
other chemistry errors and syntax failures have **no accepted identity**.
Their source index, raw SMILES SHA, candidate canonical SHA (if available) and
reason are stored in the same atomic chunk transaction. No raw labels or
label statistics enter the anomaly report.

## CPU release and recovery

Run the existing preparation CLI with `--identity-only` first. The reusable
`unified_roles_cpu.pbs` requires explicit, confined immutable staging and output
paths; identity-only is its default. Three workers plus parent use four CPU
slots, no GPU, a two-hour ceiling. Do not confuse requested capacity with spend.

The complete official-train identity pass records ALL anomalies, not one failed
retry per row. Null identities are never ranked or selected. Any unknown anomaly
leaves `ROLE_BLOCKED`, including after recovery, and no members/graph-derived
roles are published. Unknown exceptions need separate evidence review, never
automatic allowance. Complete known-only inventory yields `IDENTITIES_VERIFIED`,
**not DATA_ROLE_ACCEPTED**. The same bound preparation can then resume without
`--identity-only` to reuse identity transactions and perform existing member,
graph-content and role acceptance. Anomalies and the final identity report are
revalidated during data acceptance.

CPU fixtures cover known exceptions/aliases, unchanged normal identities,
stereo/isotope/charge distinctions, unknown errors, full audit, null-identity
selection rejection, cross-role duplicate exclusion, interrupted transaction
rollback/recovery, multi-process equivalence, ledger/source pin tampering,
strict-r2 output incompatibility and audit tampering. Local RDKit is not a
substitute for the frozen compute-node runtime or real complete-corpus proof.

## Evidence ownership

Live state is only in `CURRENT_STATE.md`. Historical hardening acceptance and
r2 failure records remain unchanged. New source/rule/CPU receipts and retrieved
audit outcomes are stored separately under the owning platform record tree.
No private mirror, indexed-input ABI or Modern2D release follows from identity
tests alone. Stop after the authorized data acceptance or an actionable block.
