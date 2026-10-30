# GPTrans PairNorm 500K initialization noise and same-recipe ensemble

Prospective desktop question, 2026-10-31. Owner: `codex/exp/pcqm-gptrans-pairnorm-init-noise-500k`.
User instruction (2026-10-31, relayed by the coordinator): research local RML and
submit new experiments mainly on kaggle2/kaggle3.

## Question

How much does the accepted 500K GPTrans BondLocal + pair-update-norm recipe
(dev 99.987 meV, run `pcqm-gptrans-pairnorm-lr-500k-s42`, arm `pairnorm_lr1e3`)
change with the initial weights alone, and what does a same-recipe mean of N
predictions gain at 500K?

## Why it decides something

- Every 500K verdict so far is single-seed and sits at 0.5-2.1 meV: PairNorm vs
  BondLocal -1.96 (descriptive), Noisy Nodes + PairNorm -2.06, PairNorm -1.14,
  LR 5e-4 +0.52, wide vs deep 0.71. No 500K noise floor exists. The 100K K1 floor
  (spread 1.63 meV over 3 inits, about 1 meV per fixed-recipe pair) is not a 500K measurement.
- The 500K ensemble curve (95.99 meV at N=4, fitted a=94.6) mixes size and LR
  variants. A same-recipe curve is unmeasured. It decides whether a deliverable
  ensemble or an out-of-fold distillation addon (about 20 trainings) is worth its cost at 500K.
- Cross-architecture: adding the best K1 500K (103.85) to four GPTrans gave -0.6 meV.
  With five same-recipe members this is re-measured on a cleaner base.

## Design

- Kernel A, kaggle2 (kaseichou): arms `init_s43` (same-run reference by role) and `init_s44`.
- Kernel B, kaggle3 (nvoid912): arms `init_s45` (same-run reference by role) and `init_s46`.
- Each arm recipe is a byte copy of the accepted reference recipe
  (`experiments/pcqm_gptrans_bondlocal_pairnorm_lr_500k/reference_recipe.json`, sha b0df767e)
  with only `initialization_sha256` changed. States come from torch seeds 43-46 through
  the same construction as `make_bond_local_model`; the seed-42 build reproduces the
  frozen state a65034da (see each `initialization_provenance.json`).
- Held fixed: 60 passes, LR 1e-3, WD 0.05, warmup-cosine, EMA 0.999, batch 128, FP32,
  sampler and dropout RNG (spec seed 42), fixed 500K train rows and 50K development rows
  (source_idx 500000-549999, inside the official train split). The seed-42 arm is not
  re-run; it is cited from its accepted terminal record.
- Measures init variance only. Full seed variance is at least this large.
- Official validation, test_dev and test_challenge are never read.

## Pre-registered reading

Singles: x42 = 99.987 (retained) and x43..x46 from the new arms (50K dev MAE, meV,
best-development-EMA selection as in the reference). D = max - min over the five;
SD = sample standard deviation over the five. E5 = mean single MAE minus MAE of the
5-member equal mean prediction. Fit a + b/N over all subsets.

- D < 1.0 meV: 500K init noise is small; past 500K single-seed verdicts of 1-2 meV stand as measured.
- 1.0 <= D < 2.0 meV: 500K gates of 3 meV stay; 1-2 meV verdicts are unresolved.
- D >= 2.0 meV: single-seed 500K screens are underpowered; future 500K screens need >= 2 inits per arm.
- E5 >= 4.0 meV and a <= 95.5 meV: same-recipe ensembling is the leading 500K lever;
  nominate a deliverable-ensemble or out-of-fold distillation question to the owner (no adoption here).
- E5 < 2.0 meV: the mixed 500K curve overstated ensembling; same-recipe ensembling has low value at 500K.
- Cross-architecture readout: MAE of 5 GPTrans + best K1 500K (late-weight-average clean,
  103.85) with equal and split-half-fitted weights, against the 5-member mean.
  A gain >= 1.0 meV keeps K1 + GPTrans mixing open at 500K; a gain < 0.5 meV closes it at 500K.
- If only one kernel finishes: report its two singles as noise samples with x42; no D threshold.

Falsifier: D < 1.0 meV and E5 < 2.0 meV together mean neither init noise nor
variance limits 500K GPTrans, and the floor is data or representation.

Row bootstrap is reported but is not training stochasticity.

## Limits

No retry, full training, adoption or protected-role use. A resumed segment needs
separate recovery authority. Ensemble numbers are offline analysis of saved
`development_predictions.pt`.

Background: Lakshminarayanan et al., deep ensembles, arXiv 1612.01474; GPTrans,
arXiv 2305.11424 (reference recipe shape).
