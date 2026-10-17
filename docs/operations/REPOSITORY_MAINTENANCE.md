# Repository maintenance

Use the [branch routing and artifact lifetime](../../BRANCHES.md#local-checkout-and-artifact-lifetime)
rule for ownership and storage. [Canonical custody](EXPERIMENT_CUSTODY.md)
records Git identities and evidence entries, not permanent local directories.

## Environment

Keep one primary Windows project environment. Install editable from the primary
checkout only. A temporary worktree selects its own source through `PYTHONPATH`:

```powershell
$projectPython = 'D:\文档\molgap\.venv\Scripts\python.exe'
$env:PYTHONPATH = (Join-Path $PWD 'src')
& $projectPython -c "import molgap; print(molgap.__file__)"
& $projectPython -m pytest
```

The printed source must belong to the selected checkout. An editable `.pth`
pointing at another experiment is an environment error, not evidence that the
selected source passed. A `.venv` junction is a shared environment alias; do not
count it as another installed copy or recursively delete its target.

## Test scope

Default pytest collection is restricted to `tests/` by `pyproject.toml`.
Package snapshots, downloaded outputs and frozen source copies are inputs, not
additional test suites. Use synthetic fixtures for infrastructure changes and
the owning model/package tests for implementation changes. Keep scientific
identity, finite-state, source-closure and provenance checks; arbitrary document
line counts are not release criteria.

Repository-layout checks inspect tracked maintained files. Frozen scientific
records keep their original hashes and paths; their integrity belongs to the
owning receipts and RML validators rather than hardcoded layout-test hashes.

## Storage cleanup

1. Inventory exact paths, file IDs/hardlinks, Git HEAD, dirty/untracked work and
   ignored payloads. Clean Git status does not prove ignored evidence is safe.
2. Verify source reachability from its reviewed Git destination. A detached
   checkout is not automatically obsolete, and an unresolved owner stays explicit.
3. Verify required non-Git bytes have a retrievable hash-bound copy. Keep sole
   evidence and active input/state until custody is established.
4. Retire a verified redundant checkout with `git worktree remove` at its checked
   absolute path. Do not use blanket `git clean`, forced pruning or recursive
   deletion across worktree/junction boundaries.
5. Measure actual free space after cleanup. Summed path sizes double-count
   hardlinks; deleting one alias may release no data blocks.

Immutable, byte-identical frozen inputs may share a verified hardlink on one
volume. Hash both copies before replacement and verify the retained path after;
record source, target and digest. These aliases must not become mutable training
outputs. Do not link active checkpoints, logs or files that will be overwritten.

Large artifact custody requires manifests and durable locations; Git preserves
tracked source/metadata, not ignored payloads. Preserve accepted scientific
records during storage migration and reconcile local availability honestly.
An inherited catalog error is a storage limitation to fix, not a reason to
download every historical `.pt` into each worktree.
