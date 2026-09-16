# Residual attribution and ablation design, 2026-09-16

The three accepted predictions were aligned on all 50,000 internal-development
rows. This analysis used no model inference and did not read official
validation, test-dev or test-challenge roles.

## Zero-training evidence

K1 and GPTrans-T were complementary rather than ordered copies of one another.
Their signed residual correlation was `0.8395`; K1 had lower absolute error on
`50.93%` of rows and GPTrans-T on `49.07%`. A fixed 50:50 blend reached
`0.100656 eV`. Five-fold source-index cross-fitting produced a stable mean K1
weight of `0.5508` and OOF MAE `0.100599 eV`, improving over K1 alone by
`0.004261 eV`. Adding EdgeState to an equal three-way blend worsened this to
`0.101103 eV`.

The accepted development graph shard was retrieved independently and matched
SHA256 `1a37dc5d458396b590044d978526822e6d1cb35df223de913a837dc7277d64c1`.
Coarse structural features did not explain the gains globally: the largest
absolute Spearman correlation between a measured feature and a paired gain was
only `0.0263`.

Useful conditional patterns nevertheless appeared:

- In the highest heteroatom-fraction quartile, K1 beat GPTrans-T by
  `0.004245 eV`.
- In the highest ring-atom-fraction quartile, K1 beat GPTrans-T by
  `0.004698 eV`.
- In the highest conjugated-bond-fraction quartile, K1 beat GPTrans-T by
  `0.003950 eV`.
- For target Gap `5.58-6.27 eV`, K1 beat GPTrans-T by `0.006796 eV`.
- In the highest target-Gap quartile, GPTrans-T beat K1 by `0.002727 eV`.
- In the lowest ring-fraction quartile, GPTrans-T beat K1 by `0.001267 eV`.

These are attribution clues, not routing rules: target Gap is unavailable at
inference, and the coarse feature correlations are too weak for an authorized
router claim.

## Minimal causal ablation

The next experiment needs only two new arms because EdgeState and K1 are
already accepted under the same V4 contract:

1. `edge_local_only`: the shared 192-channel, nine-layer persistent EdgeState
   local backbone with all global communication removed.
2. `edge_sparse_global_369`: the same local backbone with standard dense GPS
   global attention only at layers 3, 6 and 9.

The existing arms complete the four-cell comparison:

| Arm | Global operation | Layers |
|---|---|---|
| EdgeState | dense GPS attention | all 9 |
| `edge_local_only` | none | none |
| `edge_sparse_global_369` | dense GPS attention | 3/6/9 |
| K1 | one molecular slot | 3/6/9 |

Interpretation is frozen before training:

- EdgeState versus local-only measures whether every-layer global attention is
  helpful or harmful.
- Sparse-global versus local-only measures the value of sparse dense global
  communication.
- K1 versus local-only measures the value of the molecular slot.
- K1 versus sparse-global separates the slot bottleneck from communication
  frequency.

Both new arms must reuse the exact matched-500K V4 data, seed, FP32/no-TF32,
physical BS128, optimizer, schedule, 60 epochs and exposure. Run them together
on isolated T4 devices. The predeclared materiality floor remains `0.003 eV`;
no additional architecture is authorized until this four-cell attribution is
accepted.
