# Matched-500K module attribution decision, 2026-09-18

## Accepted result

The exact accepted EdgeState and GPTrans-T predictions were aligned to the
accepted 50,000-row internal-development topology shard. No encoder inference
or training ran, and no official validation, test-dev, or test-challenge role
was read.

| Model | Development MAE |
|---|---:|
| Dense EdgeState GPS9 | `0.111348806 eV` |
| GPTrans-T core | `0.106867528 eV` |

GPTrans-T improved the matched EdgeState result by `0.004481278 eV`. The
advantage was broad rather than a molecule-class effect: it remained positive
in all 71 bins formed from 22 topology, chemistry-category, shortest-path, and
RWSE features. Bin-average gains ranged from `0.002153` to `0.007448 eV`.

The two signed residuals were still correlated (`0.84754`), but their top-1%,
top-5%, and top-10% hard-set Jaccard overlaps were only `0.515`, `0.496`, and
`0.508`. Observable graph summaries did not explain that remaining
complementarity: five-fold source-index cross-fitting reached only `0.5336`
winner AUC and `0.00215` gain-regression R2. The largest univariate absolute
Spearman correlation was `0.0316`.

The deleted Stage-5 K1 kernel left no retrievable accepted per-row payload. A
later 40-epoch K1 run had almost the same scalar MAE but different row
residuals and was correctly excluded. K1 conclusions in this decision therefore
come only from the already frozen exact analysis: K1 reached `0.104860 eV`, its
signed residual correlation with GPTrans-T was `0.8395`, their fixed 50:50
blend reached `0.100656 eV`, and the prior structural analysis also found only
weak global feature/gain correlation (`max |rho| = 0.0263`).

## Module attribution

The combined evidence rules out simple structure routing. It also strengthens
the earlier causal result that dense node-to-node global attention is the wrong
operation for this backbone: GPTrans-T beats EdgeState across every measured
regime, while K1's low-bandwidth molecular slot beats both EdgeState and the
local-only ablation.

Two newer V5 mechanism results, read as experimental data without importing
the server branch's contract, identify the repair more precisely:

- K1 PairToken added a normalized, learned all-pair relation bottleneck and
  improved the 100K K1 reference by `0.003044 eV`. Its frozen causal audit found
  learned selection, cross-node pairs, and per-pair channel normalization all
  necessary. Explicit all-pair construction reduced throughput substantially.
- GPTrans `pair_update_norm` normalized each new relation update before adding
  it to persistent pair memory and improved its 100K reference by
  `0.004843 eV` with no added parameters. Normalizing the accumulated pair
  state after addition regressed by `0.001987 eV`.

The bottleneck is therefore **selective cross-node relation extraction with
bounded update injection**, not parameter count, another dense attention layer,
or a coarse molecular router.

## Literature-directed repair

The next implementation should follow the evidence, not copy an entire paper:

- [Set Transformer](https://arxiv.org/abs/1810.00825) uses learned inducing
  points to reduce set attention from quadratic to linear complexity.
- [Perceiver](https://proceedings.mlr.press/v139/jaegle21a.html) uses asymmetric
  cross-attention to distill large inputs through a small latent bottleneck.
- [NormFormer](https://arxiv.org/abs/2110.09456) supports normalizing/scaling a
  generated attention update rather than repeatedly renormalizing all stored
  residual history.
- [ReZero](https://proceedings.mlr.press/v161/bachlechner21a.html) supports a
  zero-initialized residual return path, which both K1 and PairToken already use.
- [Exphormer](https://proceedings.mlr.press/v202/shirzad23a.html) is supporting
  evidence that sparse global communication can retain long-range reach at
  linear cost, but its random expander edges are not justified here because
  learned pair selection was causally necessary.
- [GPS++](https://arxiv.org/abs/2212.02229) confirms that PCQM4Mv2 benefits from
  carefully controlled local/global/edge information flow, but its 3D,
  denoising, depth, and ensemble package cannot be credited to one module and
  is outside this falsifier.

## Next actions

1. **GPTrans P0:** run one matched 500K V5 bridge for `pair_update_norm`. It is
   the cheapest qualified candidate: parameter-free, already above the 100K
   gate, and directly addresses the diagnosed relation-state update problem.
2. **K1 P1:** design one `InducedPairToken` 100K V5 falsifier. At layer 6, use a
   fixed small set of learned source/target inducing summaries, form only their
   cross relations, preserve per-pair channel normalization and learned
   selection, and return through a zero-initialized projection. Normalize the
   generated relation update before residual addition. This must remain
   subquadratic in atom count and clear both `0.003 eV` versus K1 and a frozen
   throughput floor; otherwise close it.
3. Do not train a graph-feature Router, restore dense attention, add geometry,
   or launch another full-scale blend from this result. The existing full-scale
   K1/GPTrans blend remained behind the accepted full EdgeState reference.

Machine evidence is in `analysis.json`; the frozen procedure is in
`protocol.md`. Source-branch evidence identities are
`e898e33` (`pair_update_norm`) and `629c661` plus its causal-audit history
(`K1 PairToken`).

