# PCQM4Mv2 unified cross-scale roles V2

This prospective data version reuses IMS accepted official-train rows and
topology. It does not replace historical fixed datasets, historical model
identities or V5. Source/role acceptance and platform mirrors are independent.
The [hardening acceptance](CROSS_SCALE_V1_HARDENING_ACCEPTANCE.md) and later
[data acceptance](PCQM_UNIFIED_TWO_P_ACCEPTANCE.md) retain their separate dated
outcomes; live status is only in `CURRENT_STATE.md`.

## Frozen selection

`pcqm_unified_roles.py` is the indexed-role extension of the existing fixed-data
owner. Before selection it freezes `rule.json`, existing role/manifest inventory,
source and acceptance raw-byte SHA, generator SHA, source commit, RDKit version,
seed 42, target counts and algorithm. No prediction/MAE input exists.

Preserve official-train prefixes of 100K, 500K and 1M. Exclude these members and
all documented historical development roles when selecting new Common-Dev50K
and Independent-Audit50K. The inventory maps ambiguous historical development
to the conservative union of fixed roles, never to an assumed clean subset.
Unresolved role names block generation. Official validation/test rows are never
selected. The CSV reader stops before the first non-training row in a mixed
source shard.

Canonical molecular identity uses RDKit canonical **isomeric** SMILES, retaining
stereo/isotopes/charges and clearing atom-map IDs. No tautomer, protonation or
scaffold splitting is introduced. Unparsable identities block, not replace rows.
The user-authorized compatibility revision is described in
[the Si identity repair](PCQM_UNIFIED_IDENTITY_REPAIR.md); historical strict-v1
rules retain their original behavior and bytes.
Hash `sha256("sha256-molecule-group-rank-v1:42:" + molecule_sha256)` ranks whole
unique molecule groups. A group touching an excluded/training row is ineligible.
Choose the minimum source index as its representative. The first 50K eligible
groups supply Common-Dev; the next 50K supply Audit. Persist each role in
ascending source-index order. Freeze this rule before any model scoring.

Full-minus-holdout removes **all copies** of both new held-out molecule groups,
not merely their chosen representatives. Its actual count is measured, not
assumed to be 3,278,606. Small train prefixes remain nested and unchanged.
Existing consumed development outside the preserved train prefixes is excluded
from selection; any separately forbidden future member list requires a new
versioned rule, never an automatic relaxation.

## Assets and protection

Read only accepted `store_assets.source` train rows and train-role
`store_assets.topology` chunks. Geometry/ETKDG/MMFF and official validation graph
shards are not selected. Require actual atom9/bond3/RWSE16 and reject unknown or
3D fields. Repack existing graph tensors into separate Full-minus-holdout,
Common-Dev and Audit payloads. This is filtering/collation, never graph building.
Nested training scales reference explicit offsets in safe filtered chunks;
non-contiguous members are never represented as a contiguous source interval.

Each manifest binds version, parent source acceptance, member/order/target/
topology digests, content bytes, role/history pins, generator and checks.
Atomic per-source SQLite transactions and graph-chunk receipts permit bounded
recovery. A final acceptance is published only after independent content and
membership revalidation. Old assets are read-only.

IMS preparation/identity storage is private (mode0700, umask077). Audit labels
stay only in its separate payload directory; no label statistics are reported.
Normal training/development exports must omit the Audit payload, its labels,
private identity SQLite and original Full shards. `authorize_role` denies Audit
scoring even with caller-provided frozen flags: a future owning one-time RML
role transaction is still required. This preparation is not scoring authority.
Common-Dev is selection-used, not an independent test. Historical Full models
trained on all official-train rows are ineligible for an unseen-Audit claim.

## Platforms and future handoff

IMS is the sole member-selection authority. Kaggle1/2 private exports never
resample. A mirror binds the same member/order/label/topology/version identities;
changed physical packaging additionally requires decoded content equivalence.
Logical identity equality alone is not byte acceptance or publication success.
No public dataset or ordinary audit-label mirror is authorized.

Existing graph/family training loaders currently require contiguous fixed-V1
roles. This phase intentionally does not change that frozen ABI to make V2
executable. A future reviewed indexed-input profile must consume these accepted
explicit maps, preserving role guards, before any real model training release.
No fallback to old development can fill this gap.

## Future comparison draft (non-executable)

New 100K/500K/1M runs train independently from frozen initial states, use the
same Common-Dev IDs/labels/order/MAE-eV/selection rule, and retain optimizer
steps, actual presentations, unique members observed, passes, actual LR,
Live/EMA/BN semantics, train-only target transform and native costs.
Same-step scale comparisons freeze full LR trajectory, batch128/tail/
accumulation/precision and exposure. Per-system tuned recipes require symmetric
predeclared budgets for new families and K1/GPTrans, or old scores stay context.

100K diagnoses mechanics/resources/curves; it cannot automatically reject500K
on quality. 500K and finalist seed confirmation each need independent authority
and budget. At least two finalist seeds are reserved in a later approved plan,
not launched here. Full-minus-holdout and final official/Audit evaluation remain
Desktop-owned. Modern2D implementation gates from the user's second-stage
specification remain inactive until their required actual acceptance exists.
Preparing1M/Full identities does not raise the Server500K training ceiling.

**NOT_AUTHORIZED_FOR_SCIENTIFIC_TRAINING**.
