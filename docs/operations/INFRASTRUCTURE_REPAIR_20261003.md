# Desktop infrastructure repair, 2026-10-03

## Lossless source recovery

The K1 reference [sidecar](../../experiments/v5_legacy_evidence_migration/k1_v4_100k_reference/trace_migration.json)
requires the original raw trace with SHA256
`a86d940044f0e591f5c4ac2978b1b32194d35f1bac19592195cacead1d024923`.
It was not found in the integration checkout or the inspected retained worktrees.
Desktop import `dcdda03518a281c2b06a92e374991883f4f737d4` added the reference
sidecar without retaining its repository-local raw-source dependency. The exact
path has no file history in the inspected local Git refs. This identifies a
selective-import closure gap; why earlier ignored disk copies disappeared is unknown.
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
Portable HEAD custody passed after the recovered file and reviewed repair were
committed; working-tree presence alone was not treated as portable acceptance.

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

## Verification

Repair commit: `64ba91f19fba10d3b5b616e7f6405e6bae7f61a0`.
The 25-file local regression selection passed: 1285 passed, 9 skipped.
The final source-byte, Git-attribute, index-escape and AST-guard checks passed
again in a focused selection: 7 passed. Skips include unauthorized real package /
shard qualification and unavailable Windows symlinks; these are not GPU certificates.
The final live-document edits passed the navigation selection again: 56 passed.
`research_memory check --frozen --portable` passed on the committed repair.
The Git blob retains the exact source SHA and 12489 bytes. Derived outputs,
accepted scientific conclusions, protected roles and replay exclusions are unchanged.

Both local reuse/Kaggle skills passed their skill-validator checks. Their edits
live outside the repository under the user's Codex skills directory; a Git push
does not distribute those local skill files. No credentials were modified.
This record grants no training, role access, replay qualification or promotion.
