# Track B Final Decision

## Decision

Track B research was frozen on 2026-07-30 as a PCQM Gap-only specialist. The
selected architecture is:

- tuned GPS9-192;
- tuned GPS11-160;
- fixed primary SchNet-176/160/6;
- fixed augmented SchNet-176/160/6;
- augmented-SchNet prediction plus a `+-0.10 eV` bounded correction from all
  four frozen embeddings;
- equal prediction ensemble of fusion seeds 42, 43, and 44.

It is not a general B3LYP model and cannot replace Track A.

## Accepted Evidence

Architecture and identity were selected on scaffold development before the
fixed official-validation subset was read. On 4,981 aligned rows:

| Model | Gap MAE (eV) |
|---|---:|
| GPS9-192 | 0.181596 |
| GPS11-160 | 0.181966 |
| Primary SchNet | 0.126200 |
| Augmented SchNet | 0.122450 |
| Three-seed bounded Fusion | **0.112011** |
| GINE v7 reference | 0.184618 |

The selected Fusion improved the augmented-SchNet identity by `0.010438 eV`
and the GINE v7 reference by `0.072607 eV`.

## Asset Acceptance

The local custody copy contains all four encoder checkpoints and all three
selected fusion heads. Their SHA256 values match `metrics.json`. The payload
manifest and fusion completion manifest also match:

- payload manifest:
  `b9b8106e4558d4d3d1988dd43923bfac2b3b21cd4f15408a6e960f4f2bb0939a`;
- fusion manifest:
  `483972e13b50b942f476cbcc1ce7b99706e235e79416a23302c705f6dc10abe1`.

## Claim Boundary

The accepted number is a fixed official-validation proxy, not an OGB
leaderboard score. Official test and sealed 20K were not used. No further
official-validation tuning, encoder retraining, fusion search, or added seed is
authorized before the presentation.

Remaining work is limited to packaging the seven selected checkpoints,
recording inference cost, running a reproducible local inference smoke test,
and preparing slides.

Source evidence:

- `experiments/pcqm_route_b/results/official_valid_5k_fusion/decision.md`
- `experiments/pcqm_route_b/results/official_valid_5k_fusion/metrics.json`

