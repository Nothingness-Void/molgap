# Correction: terminal_decision.md hash mismatch, 2026-10-31

`experiments/pcqm_k1_init_noise_100k/terminal_decision.md` no longer matches the SHA-256 that the closure
records bound at finalization.

| | SHA-256 |
|---|---|
| Recorded at closure (in 18 files below) | `28972c6c4b442fdaa9062bfa3ca94c2c924188f17addac1c86f094f39a85767d` |
| File as committed in `c93e0ad6` (2026-10-30 16:00 JST) and today | `20300acf402ca7b82fc5784251524b6417e253130feb8c917035e00b2d88f178` |

## What is known

- The file was edited after the finalization records captured the old hash and
  before commit `c93e0ad6`. No other version exists in git history.
- The original bytes were searched for and not found: all git objects of this
  repository (including stashes and the objects of the seven deleted local
  experiment refs, which are still in the object store), the artifact root, the
  scratch root and the retained worktrees. Nothing hashes to the recorded value.
- The content of the original is therefore unknown. This note does not claim what
  changed or that the numbers are unchanged.

## What is not changed

- No closure, finalization, terminal or qualification record was edited, and no
  hash was rewritten to match the current file. The stale hash stays in those
  records as evidence of what was bound.
- The measurements and outcomes (ref_s42 CLOSED, init_s43 and init_s44
  NEGATIVE_UNDER_CONTRACT) come from the retrieved arm outputs and strict
  qualification records, not from this markdown file. The current file is read as
  the scientific narrative written at closure time, with the reservation above.

## Effect on checks

`research_memory validate` keeps failing with `bound artifact SHA mismatch` for
this file. The failure is expected until a reviewed re-binding decision exists.
`check --frozen` does not read this artifact and is unaffected.

## Records that bind the old hash

- `experiments/pcqm_k1_init_noise_100k/runs/pcqm-k1-init-noise-100k-s43/prospective/init_s43/rml_finalized/finalization.json`
- `experiments/pcqm_k1_init_noise_100k/runs/pcqm-k1-init-noise-100k-s43/prospective/init_s43/rml_finalized/terminal_input.json`
- `experiments/pcqm_k1_init_noise_100k/runs/pcqm-k1-init-noise-100k-s43/prospective/init_s43/terminal_staging/v1/eeb33fbe18588426/terminal.json`
- `experiments/pcqm_k1_init_noise_100k/runs/pcqm-k1-init-noise-100k-s43/prospective/ref_s42/rml_finalized/finalization.json`
- `experiments/pcqm_k1_init_noise_100k/runs/pcqm-k1-init-noise-100k-s43/prospective/ref_s42/rml_finalized/terminal_input.json`
- `experiments/pcqm_k1_init_noise_100k/runs/pcqm-k1-init-noise-100k-s43/prospective/ref_s42/terminal_staging/v1/aec1f2246b709c90/terminal.json`
- `experiments/pcqm_k1_init_noise_100k/runs/pcqm-k1-init-noise-100k-s43/prospective/strict_qualification/0b8872da28e60f70d874/init_s43/comparison_readiness.json`
- `experiments/pcqm_k1_init_noise_100k/runs/pcqm-k1-init-noise-100k-s43/prospective/strict_qualification/0b8872da28e60f70d874/init_s43/qualified_terminal.json`
- `experiments/pcqm_k1_init_noise_100k/runs/pcqm-k1-init-noise-100k-s43/prospective/strict_qualification/0b8872da28e60f70d874/ref_s42/qualified_terminal.json`
- `experiments/pcqm_k1_init_noise_100k/runs/pcqm-k1-init-noise-100k-s44/prospective/init_s44/rml_finalized/finalization.json`
- `experiments/pcqm_k1_init_noise_100k/runs/pcqm-k1-init-noise-100k-s44/prospective/init_s44/rml_finalized/terminal_input.json`
- `experiments/pcqm_k1_init_noise_100k/runs/pcqm-k1-init-noise-100k-s44/prospective/init_s44/terminal_staging/v1/f02c68649d6986ef/terminal.json`
- `experiments/pcqm_k1_init_noise_100k/runs/pcqm-k1-init-noise-100k-s44/prospective/ref_s42/rml_finalized/finalization.json`
- `experiments/pcqm_k1_init_noise_100k/runs/pcqm-k1-init-noise-100k-s44/prospective/ref_s42/rml_finalized/terminal_input.json`
- `experiments/pcqm_k1_init_noise_100k/runs/pcqm-k1-init-noise-100k-s44/prospective/ref_s42/terminal_staging/v1/ba3c1218534aea28/terminal.json`
- `experiments/pcqm_k1_init_noise_100k/runs/pcqm-k1-init-noise-100k-s44/prospective/strict_qualification/0da48d66475b327a96a2/init_s44/comparison_readiness.json`
- `experiments/pcqm_k1_init_noise_100k/runs/pcqm-k1-init-noise-100k-s44/prospective/strict_qualification/0da48d66475b327a96a2/init_s44/qualified_terminal.json`
- `experiments/pcqm_k1_init_noise_100k/runs/pcqm-k1-init-noise-100k-s44/prospective/strict_qualification/0da48d66475b327a96a2/ref_s42/qualified_terminal.json`
