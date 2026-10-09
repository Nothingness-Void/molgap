# Evidence review and decision question — 2026-10-07

Question: does the dropout disagreement penalty still improve the identical
pretrained two-forward K1 recipe at 500K? The user authorized this ablation
after the scale attribution review; the intended decision is how to improve
K1 at 500K, with no full-data training.

Canonical desktop evidence:

- `pcqm-k1-dropout-consistency2-kaggle3-100k-s42-v1-terminal` is the accepted
  100K consistency result.
  The matched numerical gain over mean2 is 1.5288 meV, below its frozen 3 meV
  promotion threshold. It does not prove the same gain at 500K.
- `experiments/pcqm_k1_500k_bn_calibration/terminal_decision.md`: frozen BN
  recalibration recovered 1.2451 meV in the retained 500K predictor. This
  establishes a state effect, not the cause or an independent module gain.
- `experiments/pcqm_k1_500k_bottleneck_diagnostic/terminal_decision.md` and
  `pretraining_lineage.json`: retained 100K pretraining covers 20% of 500K
  membership; clean tail-fit changes are heterogeneous; capacity saturation,
  universal exposure shortage and expanded-pretraining gains are unproven.
- Retained 500K source/evidence owner `031a890b4d604fc4614cde2da10e6e76cf2f836e`,
  `experiments/pcqm_k1_gptrans_package_transfer_500k/terminal_acceptance/`:
  teacher-free K1 reached 0.104904077 eV after all 60 passes, with best epoch49.
  Old plain K1 0.104859870 eV is contextual, not a matched penalty ablation.
- Retained 100K combo owner `51bc61fc28691fa771c50e231ee86eb8a4e1c42a`:
  teacher increment 1.4846 meV is matched; no 500K fusion teacher exists, and
  this question does not consume one or claim a full strongest-package test.

Hypothesis: the penalty may retain a small benefit, or become neutral/harmful
at 500K. Its effect cannot be inferred from a package-versus-history endpoint.
Hold initialization, architecture, two forwards/RNG consumption/BN update
count, optimizer, schedule, target transform, rows and exposure fixed. The
cheapest implementation falsifier is CPU synthetic coefficient isolation and
all-arm training-only native T4 qualification. Frozen-checkpoint intervention
cannot reproduce the optimization counterfactual, so the user-authorized
matched training pair is the required scientific discriminator.

If mean2 wins, removing this regularizer is a candidate improvement; if the
penalty wins, retain it and close this proposed explanation. Below the material
gate or incomplete evidence, do not claim causality or adopt a model. Report
both raw and fixed-recipe BN-calibrated comparisons before proposing another
module. No automatic width, pretraining-budget, teacher or full-data successor.
