# Promotion and slim, 2026-11-17

Master holds the minimal runnable unit and our results. Everything else is in
the `archive` branch and a tag. Nothing was lost.

## What master keeps

- `src/molgap/`: 12 modules, all that the public inference API imports.
  `__init__`, `compat/__init__`, `compat/legacy_constants`, `constants`,
  `fusion`, `gps`, `graphs`, `inference`, `multi2d_router_fusion`, `schnet`,
  `tensornet`, `utils`.
- `tests/`: 6 test files plus `conftest.py` (28 tests), all for those modules.
- Results: `CURRENT_STATE.md`, `production/`, `experiments/` and `platforms/`
  result files (`.json`, `.md`, `.txt`, figures, logs), `models/` notes.
- Scripts that still run against the 12 modules.

## What the slim removed (650 files, 6.0 MiB)

| Group | Files |
|---|---|
| `src/molgap/` modules and packages the inference API does not import (older models such as `hybrid.py`, `late_router.py`, `multi2d_data.py`, `schnetpack.py`, `visnet.py`, `archive/`, and the K1/GPT-T research library) | 298 |
| Tests of those modules and of removed scripts | 130 |
| Scripts in `production/`, `experiments/`, `platforms/` that import a removed module or call a removed script | 222 |

No result data was removed.

## Get it back

- Tag `master-pre-slim-20261117` (commit `b38e4ba1`) is master before the slim.
  One file: `git checkout master-pre-slim-20261117 -- <path>`.
- Branch `archive`, folder `from-master-pre-slim-20261117/`: the removed files
  at their original paths, with a README. Local only until pushed.
- K1 and GPT-T code: also on `molgap-desktop` (`e53120da`) and the experiment
  branches below.
- Restore a removed script together with the `src/molgap/` files it imports.

## Where Track B evidence lives

- `codex/exp/k1-muon-500k-20261106`, `6b824342`
- `codex/exp/gptrans-full-levers-20261115`, `33ab4571`
- `codex/exp/gptrans-full-minus-tail-20261110`, `e32009ae`
- `codex/exp/gptrans-tokengt-bond-100k-20261117`, `1dc9c014`
- `codex/exp/gptrans-width-500k-20261117`, `532be162`, `ac983785`
- K1 Muon 100K decision: desktop `5039fa0c`.
- The TokenGT and width branches also change `src/molgap`
  (`gptrans_tokengt_bond.py`, `gptrans_variants.py`,
  `kaggle_accelerator_push.py`); those changes are not on master.

## History: the first promotion

Earlier today `b38e4ba1` copied the whole desktop library (`e53120da`) onto
master. The slim above supersedes it. Desktop state stays at
`git show molgap-desktop:<path>`.

## Verification

`PYTHONPATH=src pytest` passes (28 tests). `import molgap` and the README
inference imports work. A real repaired-2M inference ran on 3 SMILES.
