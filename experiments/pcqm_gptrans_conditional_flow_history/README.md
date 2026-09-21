# Conditional-flow historical arm views

These records are retrospective views of one completed joint experiment. They
are not independent prospective experiments or new terminal finalizations.
The original joint trajectory and its `rml_finalized/` directory are unchanged.

| Arm | Historical trajectory ID |
|---|---|
| conditional_pair_readback | TH-gptrans-conditional-flow-100k-s42-conditional_pair_readback |
| conditional_pair_recurrence | TH-gptrans-conditional-flow-100k-s42-conditional_pair_recurrence |

Both records retain the actual joint run ID
`nothingnessvoid/molgap-gptrans-conditional-flow-s42:v2`. The source trajectory is
`TC-gptrans-conditional-flow-100k-s42`; the source evidence is
`pcqm-gptrans-conditional-flow-100k-s42`. Immutable source records were copied
from local Git commit `ea6240e18028663893587fbfa29cdb8090db9ccb` into `sources/`
with noncanonical filenames, so discovery does not register a duplicate joint
trajectory/evidence or attempt to validate another finalization directory.

## Existing schema representation

`trajectory-v1` already supports `record_mode=retrospective_partial`,
`owner=historical`, an action `type`, and an open `result` object. Each record
uses `actions[0].type=derived_from_joint_run`; its `result.provenance` identifies
the arm, joint trajectory/run/evidence/finalization IDs, source commit, paths,
and exact SHA256 bindings. `trace-v1` already supports an array of provenance
objects; the same relationship is attached to each canonical trace. No schema
or validator was changed. This is descriptive provenance, not a new enum or a
new authority-bearing field interpreted by the compiler.

The relationship deliberately lives in result provenance, not
`state_at_start.parent_trajectory_ids`: it was established after completion,
not frozen before an independently planned arm experiment. The empty prior
state fields do not claim independently frozen knowledge. The joint record is
not registered in the Desktop corpus, so its evidence ID is retained in
provenance with a local byte snapshot rather than falsely registered as a new
arm evidence ID. `result.evidence_ids` is empty; `result.evidence_refs` supplies
the original joint evidence/acceptance snapshots. No new V5 envelope is minted.

## Trace and authority limits

The generator `experiments/_scripts/recover_conditional_flow_history.py` reads
only local metadata. Each raw arm trace must match the hash already recorded
in the original finalization receipt. Epoch, learning rate, live training MAE,
EMA development MAE and elapsed wall interval are copied directly. The frozen
GPTrans contract selects EMA development weights; source commit `d6ca062`
in `src/molgap/pcqm_gptrans_v4.py` loads EMA weights in `_evaluate`.
Per-epoch optimizer/sample increments remain in the raw trace; this import
does not relabel them as cumulative coordinates. Canonical step/sample axes,
live-dev metric, checkpoint identities and device/cumulative times remain null.

Both arm outcomes are the negative result explicitly recorded in the retained
joint acceptance and decision, not a newly calculated scientific conclusion.
No joint cost or role events are copied or split, and no absent event implies
zero cost or untouched roles. These views have no prospective authority,
decision-state snapshot, readiness package, or new finalization. No replay
manifest is registered by this history-only import; automatic replay admission
is not claimed, and the derived corpus remains untouched.

Tests, RML acceptance/rebuild, training, inference, remote jobs and protected-role
access were not run. Luna owns validation. Snapshot byte preservation is explicit
in `.gitattributes`; original records are never rewritten by the generator.
