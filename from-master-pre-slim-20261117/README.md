# Removed from master on 2026-11-17 (slim)

These 650 files left `master` when it was cut down to the minimal runnable
unit. They sit here at their original paths, byte for byte.

- Source tag: `master-pre-slim-20261117` (master before the slim).
- Source commit: `b38e4ba1` (same commit as the tag).
- Slim branch: `codex/slim/master-20261117`.

## What is here

| Group | Files |
|---|---|
| `src/molgap/` modules and the `archive/`, `common/`, `data/`, `models/`, `training/`, `research_memory/` packages that the public inference API does not import | 298 |
| `tests/` for those modules and for removed scripts | 130 |
| Scripts under `production/`, `experiments/`, `platforms/` that import a removed module, or call a removed script | 222 |

Master keeps 12 modules: `__init__`, `compat` (2), `constants`, `fusion`, `gps`,
`graphs`, `inference`, `multi2d_router_fusion`, `schnet`, `tensornet`, `utils`.
All result data (`.json`, `.md`, `.txt`, images, logs) stays on master.

## Get a file back

Single file, original path:

```
git show master-pre-slim-20261117:<path> > <path>
git checkout master-pre-slim-20261117 -- <path>
```

The same bytes are here: `git show archive:from-master-pre-slim-20261117/<path>`.
The old `src/molgap/` K1 and GPT-T research code also lives on `molgap-desktop`
and the experiment branches.

Whole old tree: `git worktree add <dir> master-pre-slim-20261117`.

Many removed scripts need the removed modules too. Restore the `src/molgap/`
files from the tag together with them.
