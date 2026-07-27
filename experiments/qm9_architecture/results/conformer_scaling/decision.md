# QM9 inference-time conformer-averaging scaling

Date: 2026-07-27

Metrics: `scaling.json`. Per-view predictions: `predictions.npz`.
Command: `scripts/architecture/qm9_conformer_scaling.py`. Protocol: `../README.md`.

## Question

The screen measured single-conformer ETKDG and a two-conformer prediction
average, then stopped at two views. On the same fusion system, replacing ETKDG
with QM9's own DFT geometry is worth -0.02428 eV, an order of magnitude more
than any architecture change screened. Does conformer averaging keep paying past
K=2, and how much of that geometry gap can it close?

## Setup

One already-trained SchNet-ETKDG encoder (seed 42) scored over six independent
ETKDG views of the test split. Views are distinct ETKDG seeds 42-47; per-molecule
seeds are derived as `(seed * 1_000_003 + source_idx)`, so each view is an
independent conformer of the same molecule. No encoder was trained.

Views were built once on Kaggle CPU (711-768 s each, 9,542-9,557 of 10,000 rows
succeeded per view) and the curve computed locally in 69 s. Scoring uses the
exact all-view intersection, **9,482 rows**, so every K covers identical
molecules. Deltas carry a 2,000-draw paired bootstrap CI.

Single-view spread is small: mean 0.09590, std 0.00036 eV across the six views,
so the gains below are not view luck.

## Result

| K | HOMO | LUMO | Gap | average | delta vs K=1 | 95% CI | significant |
|---:|---:|---:|---:|---:|---:|---|---|
| 1 | 0.08178 | 0.09275 | 0.11352 | 0.09601 | — | — | baseline |
| 2 | 0.07927 | 0.09016 | 0.10994 | 0.09312 | -0.00289 | [-0.00332, -0.00248] | yes |
| 3 | 0.07835 | 0.08877 | 0.10882 | 0.09198 | -0.00404 | [-0.00452, -0.00353] | yes |
| 4 | 0.07789 | 0.08816 | 0.10792 | 0.09132 | -0.00469 | [-0.00521, -0.00418] | yes |
| 6 | 0.07744 | 0.08793 | 0.10735 | 0.09091 | **-0.00511** | [-0.00569, -0.00456] | yes |

Marginal gain per added view:

| step | per-view gain |
|---|---:|
| 1 -> 2 | -0.00289 |
| 2 -> 3 | -0.00115 |
| 3 -> 4 | -0.00066 |
| 4 -> 6 | -0.00021 |

## This is variance averaging, and it saturates

Fitting `MAE = a + b/K` gives **R2 = 0.9978** with asymptote **0.08989 eV** —
textbook independent-noise averaging. Extrapolating to infinite views buys only
another -0.001 eV beyond K=6, so K=4-6 already captures nearly all of it.

The consequence matters more than the number: averaging removes ETKDG's
**random** conformer noise, not its **systematic** offset from a relaxed
geometry. Of the -0.02428 eV geometry gap, only about a fifth is reachable by
sampling more conformers; the remaining four fifths is geometry quality.

## Relative leverage

| intervention | average-MAE gain |
|---|---:|
| ETKDG -> DFT geometry | -0.02428 |
| conformer averaging K=1 -> 6 (6x inference) | -0.00511 |
| second GPS expert (GPS7 -> GPS11) | -0.00115 (inside encoder-seed noise, see `../encoder_seeds/decision.md`) |
| gate width 192 -> 256 | -0.00048 |

## Conclusion

Two usable findings:

1. **Conformer averaging beats every architecture change screened** — 4x the
   second-expert margin, 10x the gate-width gain, and it trains nothing. K=4 is
   the practical point (-0.00469 at 4x inference cost); K=6 adds only -0.00042
   more. Worth adopting wherever inference cost is amortised, as in an offline
   database build.
2. **It is not the fix for geometry.** The 1/K saturation shows the dominant
   geometry deficit is systematic. Closing it requires a *better single
   geometry* (xTB or NNP relaxation), not more ETKDG samples. This raises the
   priority of the geometry backlog item that was previously ranked LOW.

Scope: QM9 small molecules, one SchNet encoder, prediction-level averaging.
Whether the same curve holds for the larger PubChemQC molecules, or for
embedding-level rather than prediction-level averaging, is untested.
