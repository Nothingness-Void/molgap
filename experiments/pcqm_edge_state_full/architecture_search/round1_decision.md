# Round 1 Capacity Decision

On 2026-08-27, both seed-42 capacity candidates completed the fixed
official-train-only 100K/10K development protocol. Artifacts were retrieved
from Kaggle, their hashes matched the completion manifests, and prediction
metrics were independently recomputed locally.

| Candidate | Overall MAE | Delta vs control | Radical delta | Non-radical delta | Decision |
|---|---:|---:|---:|---:|---|
| `edge96` | 0.150456 eV | +0.000394 eV | +0.002726 eV | +0.000201 eV | Reject |
| `deep160x11` | 0.150714 eV | +0.000652 eV | +0.008571 eV | -0.000003 eV | Reject |

Neither candidate met the predeclared `0.002 eV` overall-improvement gate,
and both exceeded the allowed radical regression. Seeds 43 and 44 were not
opened. Increasing EdgeState width or GPS depth was therefore closed for this
screen.

The shared failure pattern was localized: non-radical performance remained
approximately flat while radical performance worsened. This supported one
bounded follow-up that changes the categorical input stem rather than model
capacity. The follow-up protocol is in `round2_protocol.md`.

Raw attempts, including three pre-training environment failures and the two
accepted version-4 runs, are retained under
`platforms/_records/kaggle/training/pcqm_official_architecture_search_20260827/`.
Exact job identities and artifact hashes are in `submission.json`.
