# GRIT-SPSE at 500K: NEGATIVE_UNDER_CONTRACT

2026-11-28, Track B. One frozen 500K run (IMS A100, seed 42) finished and was
scored once on the tail holdout. It fails the first rule of
[protocol.md](protocol.md): tail MAE 116.205 meV is above the 104.273 meV line.
GRIT-SPSE is closed at 500K under contract. No retry, no grid. Nothing is
adopted. Closure built with `rml legacy-terminal` (no hand-written glue).

## Verdict

- Terminal word: **NEGATIVE_UNDER_CONTRACT** (Opus ruling, 2026-11-28).
  Abandonment confirmed.
- Rule: protocol "Cheapest falsifier": fail if final-EMA tail MAE > 104.273 meV
  (107.273 minus 3.0). 116.205 > 104.273. The second condition (paired 95%
  bootstrap bound against `gptrans_500k_init_s44`) is not needed and was not
  computed.
- Candidate tail MAE 116.205 meV; dev MAE (rows 500,000..549,999) 108.367 meV
  (diagnostic only; tail and dev are different scales, never mixed).
- Gap to the frozen reference 107.273 meV: +8.932 meV, about 8 times the
  1.09 meV init-noise range of the reference. Single seed, so init noise of
  this arm is unmeasured, but the gap is far outside the stated floor.
- `effect_meV`, `ci_meV` and `failure_mode` are null: there is no same-job
  reference arm, so no endpoint regression is recorded.

## Cross-platform note

The reference (GPT-T BondLocal+PairNorm, tail 107.273) ran on Kaggle T4; this
arm ran on IMS A100. Cross-platform shift is small: the J5 run (GPT-T Kaggle
s44) measured +0.035 meV across platforms, so the cross-job comparison holds
(see the DeMol chain decision; that source was quoted by the ruling and not
re-read here). The comparison stays cross-job, single seed, and cannot be
`STRICT_CAUSAL`.

## Run identity

- Frozen trajectory action `A001`, run string
  `pcqm-grit-spse-500k-20261118-s42:grit_spse:downstream` (copied from
  `prospective/trajectory.json`). That is what `legacy-terminal` received.
- Executed run: `pcqm-grit-spse-500k-20261118-r2-s42` (the r2 run; an r1 attempt is not
  documented in this directory). IMS jobs: preflight 2066230, training
  2066231. Status COMPLETE, 60 passes, 234,360 steps, selection final-EMA.
- Run spec: `run_spec.json` in this directory (run_id r2, sha256
  `a644941af1c4b077f4b5c8f61ef02d3f2340dd7bc1150a5c7b2ad5f9c29bdc00`). The
  retrieved copy has the submit helper's fills and a different hash:
  `evidence/run_spec_retrieved_r2.json`
  (`16d5cc1ae19831bac0215b3f948a2d6584c9731b9b6da118260fd50499dee430`).
  Recipe sha256 `44e84fea5c6df540f838971a57a979027737536ba8566c28c616d073b527186a`
  (repo copy and retrieved copy are equal).
- Training source commit `4fdeac8fe41fa7bf63d819403a4d0be9406c0d23`, archive
  sha256 `6cef868f7f432a13a7219e1412592dc56cca6655a6a7cc3b1d26728e92276843`.
  The trajectory froze `70d628da352bcf317b2b8ebdd3d5fa49e9612ca0` at planning;
  the two differ (build commits).
- Retrieved directory (exists):
  `D:/molgap-artifacts/retrieved/pcqm-grit-spse-500k-20261118-r2-s42`.
- Checkpoints stay out of the repo. `training/selected_model.pt` (final EMA)
  sha256 `481055d7b3e76272940ae6de20f85ebb72bb439509894551e1a0dda3521bc816`
  (recomputed locally, equals the manifest and the score file).
  Last checkpoint sha256 (manifest only, file not re-hashed)
  `1b4eed0bfb3263a82ec29d9fad0a4bf49b8c4388fc2a4af1880d2bdd2d876bd8`.
- Score file: `D:/molgap-artifacts/tail_eval_cutlass_grit_500k_20261121/tail_grit-spse.json`
  (copy in `evidence/`); per-row predictions in the `preds/` folder beside it.

## Role use (structural evidence, Opus ruling)

- `official_validation`, `test_dev`, `test_challenge`: **untouched**. The run
  read only the fixed 500K train store (rows 0..499,999); the run provenance
  records `official_valid_read`, `test_read` and `tail_holdout_v1_read` all
  false for training, and the score file records `protected_roles_read` all
  false.
- `tail_holdout_v1`: **consumed**. The final EMA model was scored once offline
  (128,606 rows). Events carry `selection_used: false`: the checkpoint was
  fixed before scoring and the score picked nothing.
- `dev_500k` (rows 500,000..549,999): **consumed** by the same offline scoring
  (diagnostic MAE only; `selection_used: false`).

## Cost

One measured wall figure: 16,820 s (4.6722 h) is the last `elapsed_s` of
`train_log.jsonl`, A100, training only. CPU hours, queue time and the preflight
stage are not measured.

## Unverified

- Cross-platform shift +0.035 meV is taken from the ruling; the DeMol decision
  was not re-read.
- The paired bootstrap against s44 was not run (not needed for the fail).
- Scoring ran on an RTX 5060 (torch 2.7.1), training on A100.
- Whether the gap comes from the mechanism or from the recipe is not
  separated; no claim is made.

## Not authorised

Retry, grid, a 100K companion, any adoption, an official-valid or test read.
