# Si identity repair: CPU acceptance and remaining authority

2026-10-16 JST. **INFRA_CPU_ACCEPTED / SI_IDENTITY_ACCEPTED** for the bounded
compatibility implementation and all 31 observed Si exceptions. **ROLE_BLOCKED**
for complete Unified V2 data: two P exceptions need separately authorized
compatibility. No DATA_ROLE_ACCEPTED, role publication, mirror, model scoring,
training, GPU job, or Modern2D implementation followed.

The [machine closure](../../platforms/_records/ims/pcqm_unified_roles_v2/identity_repair/closure.json)
pins actual source/artifact hashes and CPU tests. The
[implementation contract](PCQM_UNIFIED_IDENTITY_REPAIR.md) owns semantics.
Original r2/hardening evidence and historical V4/V5 model conclusions were not
modified. The user authorized same-class Si expansion only after per-structure
identity and accepted-graph evidence; P is not implicitly included.

## Complete real audit

CPU job `1731714.ccpbs1` read exactly 3,378,606 official-train source rows and
retained each 50K SQLite transaction. It found 33 strict-sanitization exceptions:
10 known Si, 21 additional Si and 2 P. No member was skipped, replaced or
relabeled. The audit completed all rows before returning ROLE_BLOCKED; its
null identities could never enter ranking/selection. There was no repeated
full-corpus retry per newly discovered exception.

Full [audit](../../platforms/_records/ims/pcqm_unified_roles_v2/identity_repair/molecular_identity_audit-r3.json)
SHA: `ca29528a06f3c53b094907be58d0e8349af3fbafc16565e7a3abf4f0de556732`.
This is an identity audit, not evidence that every unusual source molecule is
chemically correct or that the eventual held-out roles have passed acceptance.

## Independent graph and serialization proofs

CPU job `1732022.ccpbs1` matched accepted source/graph shards and verified
29 Si structures. Two other Si structures failed serialization round-trip:
default canonical SMILES lost the source zero-H constraint and re-reading
inferred one extra H. The gate rejected them instead of publishing a key.
Both P structures had matching graph tensors but remained outside the Si rule.
The original [partial proof](../../platforms/_records/ims/pcqm_unified_roles_v2/identity_repair/exception_graph_audit.json)
and its 29-member ledger were retained separately.

CPU job `1732093.ccpbs1` used the explicit-H v3 serialization, not repaired
chemistry. All 33 structures passed stable canonical/atom-order/H round-trip
checks and exact comparison of original atom9, edge-index and bond3 tensors.
Only the 31 Si structures became eligible for the frozen Si ledger. The
[final proof](../../platforms/_records/ims/pcqm_unified_roles_v2/identity_repair/explicit_h_v3/exception_graph_audit.json)
SHA is `6ae32541ec2eca869eaaf38efcc2765b24f10db687d8789e6d185a68d2c381d7`.
The [31-member ledger](../../platforms/_records/ims/pcqm_unified_roles_v2/identity_repair/explicit_h_v3/fallback_ledger.json)
SHA is `41cbad2edf8b38534cdb7d1fcbb553a1dfccae96b23054b43c3f6342c4f97da7`.
Local retrieval rechecked these against remote SHA, archived verifier bytes,
the complete original audit, source manifests and ledger/proof consistency.

The [frozen Si v3 rule](../../platforms/_records/ims/pcqm_unified_roles_v2/identity_repair/rule-si-v3.json)
has identity `3f99e15efeefda934bbe47628a7b2eb1e15ba646a5fc08b73817f741f7db4f92`.
It is deliberately **not executed for whole-data roles**: known unresolved P
members would still block. V2 keys/SQLite rows are not resumed under v3 semantics.
Ordinary sanitized molecular keys stay unchanged. Accepted graph bytes, source
labels, existing train prefixes and the database are unchanged.

## Remaining exceptions (no automatic scope expansion)

| Source index | Element/error | Identity/graph evidence | Release |
|---|---|---|---|
| 2539039 | P explicit valence6 | Stable explicit-H identity; atom/edge/bond tensors exactly match accepted graph | Separate P authority needed |
| 2980839 | P explicit valence6 | Stable explicit-H identity; atom/edge/bond tensors exactly match accepted graph | Separate P authority needed |

Do not add a guessed negative charge, change a bond, delete/replace either row,
or change the training prefixes to avoid these exceptions. The offered bounded
successor is a separately frozen two-P whitelist with the same evidence gates,
then complete CPU role/content acceptance. It is **not released by this record**.
No additional full inventory was submitted merely to rediscover these two rows.

## Implementation, tests and failures

Implementation HEAD `1683337a762e3b21df54acfaaf8615b189b13b75` on molgap-server;
closing docs/evidence commit is separate. No push or Desktop integration occurred.
Reused fixed-data source pins, SQLite recovery, selection, graph filtering,
PackedGraphDataset, registered runtime, atomic IO and existing OGB feature
mapping. The cache helper gained optional paired feature functions; its default
OGB behavior is unchanged and exact local parity was tested. No training loop,
optimizer, sampler, checkpoint recovery or platform submission framework was
rewritten.

Final CPU regression: **219 passed, 0 failed, 0 skipped**. Earlier source/layout/
package regression:104 passed/1 skipped (Windows symlink availability). Counts
overlap and must not be summed. RML read-only validation passed:110 evidence,
127 trajectories,193 costs,481 roles,64 traces; no canonical records or derived
research/user edits were overwritten. Compile and documentation checks passed.

Failures were retained, not suppressed: two test commands named nonexistent
files and collected no tests; a Windows output-path assertion failed (76 pass/
1 fail), then normalization and77-/219-test rechecks passed. Pre-submission
Torch/OGB import checks exposed omitted runtime dependencies; no jobs were
created by those checks. Job `1731916.ccpbs1` exited before graph access because
a guard followed the venv interpreter symlink; the embedded PBS guard itself
was tested after repair, with exact source/identity inputs in a new stage.
Job1732022 was a successful partial proof, not an infrastructure failure or
complete data acceptance. Raw logs/packages/JUnit stay ignored and SHA-bound.

## Preservation, cost and stop

Observed old fixed manifest SHA remained
`4ecd1546c5fc898004809a5b7130bd7963a5eb569120dbee95e65f18ed9cc99e` and acceptance
SHA remained `82bdf0d304b5a1b20d120dd25c07ad6c6cf02324bab2b13634ed68ee8c63db0e`.
Selected source/topology shards were SHA-verified; no complete25GB rescan or
RWSE reconstruction is claimed. No geometry or official validation/test shard
was selected. Existing graph payloads were read, never rewritten by this repair.

Each job requested four CPU slots, zero GPUs and a two-hour ceiling. Native
whole-allocation CPU/wall/queue cost was unavailable and remains null, not
four times two hours of claimed spend. No GPU allocation or training was used.
The final exact scheduler observation was empty; the authorized CPU chain
stopped. Normal training exports/Audit scoring remain closed. The user test
retained SHA `e623abd9b0a410d6c49ccf1d04e2e1ba5bb23237db4dea2a2c8c874f4b9465f1`.
