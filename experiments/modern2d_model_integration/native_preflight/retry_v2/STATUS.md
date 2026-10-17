# Bounded qualification retry

Owner: server. Authority: [authority.json](authority.json).

## Exact attempt

Kaggle2 `kaseichou/molgap-modern2d-native-100k-v2`, kernel139267796/version1,
was submitted once and observed ERROR at 2026-10-17T17:40:28.606718+00:00.
The [inspection](results/inspection_2026-10-18.md) retained exact-version outputs
and all13 manifest-bound artifact hashes. Source/data discovery and environment
setup passed; both workers failed at their first forward-output device check,
before backward or an optimizer update. Native qualification is NOT_ACCEPTED;
this is a shared execution-interface failure, not a scientific route negative.

PPGT/2 and ESA/3 use separate devices, RNG, models, optimizers and checkpoints.
The limit is T4x2,1800 seconds/1 allocated T4-hour, BS128, seed42 and six scratch
optimizer updates/model. This is numerical/resume/resource qualification only:
no 60-pass training, scientific ranking, protected roles, extra seeds or automatic
successor is authorized. Native qualification and full allocation cost are
unverified. No repair or remote retry was executed during this inspection.

## Frozen release and retrieval

The [submission record](results/submission_2026-10-18.json) binds the exact source,
Spec, package, launch, tests and measured preparation phases. Data version1,
recipes and both initial states were reused; no graph/data export was repeated.
The private source dataset `kaseichou/molgap-modern2d-qualification-source-v2`
passed all12 downloaded-file SHA checks at version1, followed by the separate
qualification-launch validation and shared release check against those bytes.

Raw operation intents/responses, immutable launch receipt, source roundtrip,
CPU logs and prepared package are retained under
`E:/MolGap/artifacts/modern2d-native-preflight/retry_v2`.
Retrieve only this kernel/version's bounded qualification artifacts for acceptance;
do not submit another version based on silence, an incomplete log or queue state.
The failed [V1](../results/decision_2026-10-18.md) is preserved as a separate
infrastructure failure, not a scientific negative.

The delegated RML validator passed. Its frozen-derived check reported the same
seven stale derived outputs retained from the user's worktree, without new-plan
errors; no derived rebuild or unrelated edit was performed.
Those were submission-time checks, not terminal acceptance. Prospective RML
snapshots remain unfinalized and must not be interpreted as live job state.
