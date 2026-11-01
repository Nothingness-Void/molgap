# K1 post-BN 100K: two 20-pass members against one 40-pass model

Prospective desktop question, 2026-11-01 JST. One kaggle2 kernel, two 20-pass arms.
Owner: codex/exp/pcqm-k1-short-ensemble-100k. This is a full-flow infrastructure
test that also answers a real question; it releases nothing beyond this kernel.

## Question

At equal training compute, does the mean prediction of two independently initialized
20-pass K1 models beat one 40-pass K1 model on the 50000 development rows?

## Evidence review (RML, 2026-11-01)

- `lookup --mechanism ensemble` returns no trajectory. `lookup --mechanism distillation`
  returns only the closed fixed-fusion distillation (both students NEGATIVE_UNDER_CONTRACT).
  No compute-matched ensemble trajectory exists, so this is not a repeat.
- [K1 100K init noise](../pcqm_k1_init_noise_100k/terminal_decision.md): three 40-pass
  inits give singles 141.2945, 139.6632, 140.6475 meV (mean 140.535). The mean of two
  members scores 134.84 meV (gain 5.70) and of three 132.85 meV (gain 7.68). That
  ensemble costs N times the 40-pass training, so it is not compute-matched.
- GPTrans PairNorm 500K init noise (owner worktree, branch
  codex/exp/pcqm-gptrans-pairnorm-init-noise-500k): E5 = 4.507 meV, nominated an ensemble
  or distillation question to the owner. No equal-compute comparison exists there either.
- Retained 40-pass predictions were re-read with
  `analyze --op ensemble-curve` (output under D:/molgap-scratch/fullflow-test/prior/):
  N = 1, 2, 3 give 140.535, 134.836, 132.852 meV.
- Unknown: how much a 20-pass cosine schedule costs one model. No 20-pass K1 100K result
  exists, so the cost of halving the schedule is the thing measured here.

## Design

- Arms `short_s43` (reference) and `short_s44`, one per T4 in one kernel. Only the
  initial weights differ: torch seed 43 and 44 (the retained `k1-init-noise-100k/s43.pt`
  and `s44.pt`; state sha256 in each `initialization_provenance.json` of the K1 init-noise
  study). Spec seed label stays 42 because the compiler permits only seed 42.
- Recipe = the retained post-BN 100K recipe with `schedule.total_units` 40 -> 20 and the
  matching `optimizer_steps` 15620, `completed_passes` 20, `sample_presentations` 1999360.
  Cosine runs to its minimum LR at pass 20. Everything else is unchanged: learning rate 4e-4,
  batch 128, EMA 0.999, Live selection on the development cohort, sampler seed 42, FP32.
- Train rows [0,100000), development rows [100000,150000). Protected roles are never read.
- The 40-pass comparator is the retained init-noise predictions (3 inits, same rows and
  targets). `ref_s42` was bit-identical across two accounts, so same-seed T4 training is
  deterministic here and cross-kernel comparison is usable. No 40-pass baseline is retrained.

## Pre-registered reading

Numbers come from `python -m molgap.research_memory analyze --op summary` on the two
accepted 20-pass prediction files (`ensemble_mae_meV`, `singles_meV`), with the retained
40-pass mean S40 = 140.535 meV and best single B40 = 139.663 meV.

- POSITIVE: `ensemble_mae_meV` <= S40 - 3.0 = 137.535 meV. Equal-compute 2-ensemble beats
  one model by the 3 meV gate. Becomes a candidate for the owner; nothing is adopted.
- NEGATIVE: `ensemble_mae_meV` >= B40 = 139.663 meV. Splitting the budget does not beat
  the best retained single model.
- Otherwise INCONCLUSIVE (137.535 < ensemble < 139.663): the 3 meV gate is unmet and the
  init spread D = 1.63 meV makes a smaller claim unresolved.
- Diagnostic only (no threshold): mean 20-pass single MAE minus S40 (cost of halving the
  schedule) and the 20-pass 2-member gain (single mean minus ensemble).

Falsifier: if the mean 20-pass single MAE exceeds S40 by more than 8.0 meV, the shorter
schedule loses more than the 40-pass 2-ensemble gain (5.70 meV) plus margin, and the
ensemble cannot recover it. The route then closes NEGATIVE without further runs.

## Limits

- Two members, one data order, one dropout RNG; seed variance is at least the init spread.
- Row bootstrap is not training variability. Each member's checkpoint is selected on the
  same development cohort, so every MAE is mildly optimistic; the ensemble mean is not tuned.
- Ensemble numbers are offline analysis of saved `development_predictions.pt`, not a
  trained or deliverable model. Inference would need two forward passes.
- No retry, continuation, 500K, full training, distillation or adoption.
