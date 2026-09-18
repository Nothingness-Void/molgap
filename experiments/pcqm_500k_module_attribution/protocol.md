# Matched-500K module attribution protocol

## Question

Which observable molecular regimes explain the accepted residual differences
among Neural-Atom K1, GPTrans-T core, and dense EdgeState GPS9, and do those
regimes support a specific module repair rather than another architecture
guess?

## Frozen evidence

- Role: the existing 50,000-row internal development role with source indices
  `500000..549999` from the accepted fixed 500K dataset.
- Models: the accepted seed-42, physical-BS128, FP32/no-TF32, 60-epoch V4
  predictions for GPTrans-T and EdgeState, plus the already frozen K1 summary
  in `../pcqm_500k_v4_evidence/residual_attribution.md` and
  `../pcqm_500k_v4_evidence/local_ablation_analysis.json`.
- The deleted Stage-5 K1 kernel left no retrievable per-row payload. A later
  40-epoch K1 run has nearly the same scalar MAE but different row residuals,
  so it is explicitly excluded instead of being substituted. New per-row
  structural diagnostics are restricted to the two exact accepted artifacts.
- Graph input: the accepted OGB-rich topology shard only. No conformers or
  geometry are needed.
- The CLI pins all four input SHA256 values and requires exact source-index and
  target alignment.

## Analysis

1. Extract graph size, degree, cycle, shortest-path, atom/bond category, and
   RWSE summary features without model inference.
2. Measure each model's conditional error, paired gains, and hard-tail overlap.
3. Rank feature/gain associations and run five-fold source-index diagnostic
   fits to estimate whether these observable features can explain the paired
   gains. These fits are diagnostics only; they do not authorize a Router.
4. Combine this result with the already accepted causal global-communication
   ablation. Search literature only for the module implicated by both bodies of
   evidence.

## Boundaries and stop rule

No encoder training or inference is run. Official validation, test-dev, and
test-challenge roles remain unread. If structural diagnostics have weak OOF
explanatory power, close feature routing and prefer a fixed low-bandwidth
fusion or representation-level repair. If one structural regime is stable and
mechanistically consistent with the causal ablation, nominate one cheapest
V5 falsifier; do not submit it from this experiment.
