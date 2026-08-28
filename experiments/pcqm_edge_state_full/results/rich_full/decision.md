# Rich-Feature Full Official Validation Decision

The one-time 14-hour IMS run completed on 2026-08-28. It trained the fixed
OGB-rich EdgeState Structural GPS from random initialization on all 3,378,606
official training rows and selected epochs only on the 73,545 official
validation rows.

The selected epoch 19 reached official-validation Gap MAE `0.102063 eV`. The
rejected legacy-feature full model reached `0.189450 eV`; restoring the complete
OGB categorical atom and bond contract therefore improved MAE by `0.087387 eV`
(`46.13%`). The full rich-feature repair passes its predetermined material
improvement gate.

All 73,545 validation predictions are finite, uniquely keyed, and ordered.
Their recomputed MAE is `0.102063 eV`. The selected and resumable checkpoints
are finite, and the completion-manifest hashes match the retrieved files.
`progress.json` retains its last epoch-level `training` state, but the later
completion manifest, metrics, and predictions prove terminal completion.

This is an official-validation result, not an official test score or
leaderboard rank. The checkpoint may proceed to the already planned
single-process raw-SMILES test inference and timing gate. No PubChemQC data,
pretrained weights, official test labels, or production registry changes were
used.

Machine-readable acceptance is `acceptance.json`; immutable retrieved evidence
is under `platforms/_records/ims/pcqm_rich_full_20260828/`.
