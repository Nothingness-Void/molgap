# Terminal identity audit - 2026-10-10

Formal finalization was NOT executed. Existing corpus validation passing does
not certify these two prospective arms as terminal or replay-ready.
[Result](result.json) retains each mismatch without changing inputs.

| Field | Frozen prospective | Physical producer / shared translator |
|---|---|---|
| Source commit |38572b120f93589e40f6632d9c281e39122c3345|a47268e54968e83c55f22196da6606d8f935bfda|
| Run ID |logical run + `:mean2` / `:single`|logical run + arm + `:downstream`|
| Attempt ID |`kaggle3-cost-quality-001`|complete descriptor emits `mean2-v1` / `single-v1`|

Both trajectory IDs agree. Prelaunch package and actual entry/source/recipes
are pinned and mechanically verified. The defect maps executable identities
to already published plans; it is not an altered model or failed GPU run.
Git comparison38572b12..a47268e5 changes only12 preparation/recipe/policy
files (45 added lines), not shared executable Python. The source-commit mismatch
is nevertheless real under the existing exact-identity gate; semantic similarity
does not permit substituting a different recorded commit.

`experiment_terminal.translate_terminal_descriptor` requires terminal run/action,
source and attempt equality; canonical trace must match the terminal run ID.
`research_memory.paired.terminal_reference_evidence` binds accepted control
source to prospective source. Closing mean2 first cannot cure these differences.

No fake terminal, reference bundle or passing comparison_readiness was published.
Prospective records stay ACTIVE pending honest closure; the negative scientific
decision is separately retained. RML does not yet contain this as a closed replay
world. Cumulative native device time and strict comparison/role evidence also
remain missing; RML validation is not terminal qualification.

Future releases should verify all three identities before submission.
Historical reconciliation needs a reviewed explicit realization/provenance path
supported by shared lifecycle, not edits to frozen prospective/source/trace,
hidden trace, bypassed validation or a substitute reference. This acceptance
does not implement that repair. Keep the owner until custody/routing is resolved.
