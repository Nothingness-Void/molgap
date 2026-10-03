# Desktop infrastructure repair, 2026-10-03

## Lossless source recovery

The K1 reference [sidecar](../../experiments/v5_legacy_evidence_migration/k1_v4_100k_reference/trace_migration.json)
requires the original raw trace with SHA256
`a86d940044f0e591f5c4ac2978b1b32194d35f1bac19592195cacead1d024923`.
It was not found in the integration checkout or the inspected retained worktrees.
One bounded, single-file recovery request to the recorded Kaggle2 kernel returned
HTTP 403. No output was downloaded and no kernel was submitted or retrained.

The retained canonical observations and original JSON writer permit lossless
byte recovery. Invert the frozen recovery field mapping, emit fields in the
original order (epoch, optimizer_steps, sample_presentations, train_normalized_mae,
development_gap_mae_eV, learning_rate, seconds), then append the writer's `improved`
flag (strictly lower than all preceding development MAEs). Serialize
`{"epochs": rows}` with Python JSON indent2 and one LF, without sorting keys.
The complete 12489-byte result matches the pre-existing source SHA256 exactly.
It was restored at the sidecar's retained_source_ref only after this comparison.
The hash, not the plausibility of reconstructed values, authorizes the restoration.
No source/config/role/cost identity, canonical trace or accepted outcome was changed.
The retained-source directory has `-text` Git attributes so a Windows checkout
cannot change LF bytes and silently invalidate the original SHA.

`research_memory check --frozen` then passed without changing derived outputs.
Portable HEAD custody still requires committing the recovered file and reviewed
repair; working-tree presence alone is not portable acceptance.

## Maintenance scope

Restore byte-preserved SSMA context documents, add owning README pointers, and
route the experiment index to the accepted accuracy decision rather than an earlier
NO_TRAIN attempt. Compact live status by moving historical navigation to its owners.
Skills point to the registered workflow, while platform submission stays in skills.

Layout tests must recognize Markdown directory links and AST main guards.
Named frozen-source allowances require both exact bytes and their accepted receipt;
they are not general exemptions for new code. Snapshot-relative historical links
remain unchanged, with hash-verified local evidence navigation from the owning README.
The unbound hydration helper uses the shared repository-root constant.

Verification results are reported by the repair task; this record grants no
training, role access, replay qualification or promotion.
