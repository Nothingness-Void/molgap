# Evidence review — 2026-10-05

RML and the canonical decisions were reviewed before selecting this question.

- [K1 dropout pair](../pcqm_k1_dropout_consistency/terminal_acceptance/decision.md):
  consistency beats matched mean2 by1.528783meV, row interval[0.616136,2.348938],
  below its3meV gate. This is a numerical clue, not adopted architecture gain.
- [Fixed equal-blend qualification](../pcqm_k1_consistency_fusion_transfer/terminal_decision.md):
  2.811987meV gain over consistency on source_idx[500000,550000), unfitted
  weights and two-model inference. Existing teacher training predictions bind
  that frozen50:50 pair. The qualification cohort is not a sealed test.
- [Direct teacher distillation](../pcqm_k1_fusion_distillation/terminal_acceptance/decision.md):
  lambda1 gains only0.432794meV with bounds crossing zero; it remains3.629717meV
  behind the teacher. Both40epoch students are NEGATIVE_UNDER_CONTRACT.
  [Attribution](../pcqm_k1_fusion_distillation/terminal_acceptance/attribution.md)
  lacks imitation/total-objective curves; longer training is not a proven fix.
- Archived pretraining acceptance at owner8821b5ce retains a completed10-pass
  local-reconstruction checkpoint. K1's historical point gain0.197185meV lacks
  strict comparator qualification; INCONCLUSIVE is preserved. Source/checkpoint
  metadata in this question bind reuse, without claiming pretraining benefit.
- FLAG,slot96,SSMA and 500K geometry/regularization do not establish material
  improvements to stack onto K1. GPTrans pair normalization is family-specific.

New question: does adding a mean-output teacher target to an identical
pretrained, consistency-regularized K1 provide transferable clean Gap gain?
This changes initialization and the interaction formulation relative to the
closed random-initialized, single-pass distillation experiment. ArmA isolates
the teacher increment. No old baseline is retrained. This pair cannot isolate
pretraining itself or establish a full-scale improvement.

Cheapest falsifier: one synthetic/interface batch and all-arm training-only
T4 qualification before formal exposure. Teacher data are cached and frozen;
no new teacher inference, geometry construction or protected-role access.
