# CutlassGT-2026 at 500K: NEGATIVE_UNDER_CONTRACT

2026-11-28, Track B. One frozen 500K run (IMS A100, seed 42) finished and was
scored once on the tail holdout. It fails the first rule of
[protocol.md](protocol.md): tail MAE 117.660 meV is above the 104.273 meV line.
CutlassGT-2026 is closed at 500K under contract. No retry, no grid. Nothing is
adopted. Closure built with `rml legacy-terminal` (no hand-written glue).

## Verdict

- Terminal word: **NEGATIVE_UNDER_CONTRACT** (Opus ruling, 2026-11-28).
  Abandonment confirmed.
- Rule: protocol "Cheapest falsifier": fail if final-EMA tail MAE > 104.273 meV
  (107.273 minus 3.0). 117.660 > 104.273. The second condition (paired 95%
  bootstrap bound against `gptrans_500k_init_s44`) is not needed and was not
  computed.
- Candidate tail MAE 117.660 meV; dev MAE (rows 500,000..549,999) 110.352 meV
  (diagnostic only; tail and dev are different scales, never mixed).
- Gap to the frozen reference 107.273 meV: +10.387 meV, about 10 times the
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
  `pcqm-cutlass-gt-500k-20261118-s42:cutlass_gt:downstream` (copied from
  `prospective/trajectory.json`). That is what `legacy-terminal` received.
- Executed run: `pcqm-cutlass-gt-500k-20261118-r2-s42` (r1 failed at the
  preflight stage and trained nothing). IMS jobs: preflight 2066226, training
  2066227. Status COMPLETE, 60 passes, 234,360 steps, selection final-EMA.
- Run spec: `run_spec.json` in this directory (run_id r2, sha256
  `46c5b06e9bd45db5acde998ded9039260c42bfcb0bb6c45b285d1e688032aa74`). The
  retrieved copy has the submit helper's fills and a different hash:
  `evidence/run_spec_retrieved_r2.json`
  (`3f332f83fcf23d0a57f6060efa88fe5331c926d7fd364cddab3259c45b99a61e`).
  Recipe sha256 `9ef9540469f912b5de6af0cc20cd8f6c7f81bfe675ce885dad7b0202a4773d15`
  (repo copy and retrieved copy are equal).
- Training source commit `ed6005c2fb931d68265e0b8e26144ffaccd70dc3`, archive
  sha256 `84fe3f9f2894e2d7fc938686958f73fa13062baf88a71b55c6faefb256d4b167`.
  The trajectory froze `c9fa14bf75581257630fe8e52a1c8c5e82621869` at planning;
  the two differ (build commits).
- Retrieved directory (exists):
  `D:/molgap-artifacts/retrieved/pcqm-cutlass-gt-500k-20261118-r2-s42`.
- Checkpoints stay out of the repo. `training/selected_model.pt` (final EMA)
  sha256 `8e41965f0c1335b8336de089ff308bda8b6d29252742ed16650305b7f26ca0dd`
  (recomputed locally, equals the manifest and the score file).
  Last checkpoint sha256 (manifest only, file not re-hashed)
  `7128a3a1355f1f2ee3cde9118546e421bd6ca38e85198e9b70646debbae92739`.
- Score file: `D:/molgap-artifacts/tail_eval_cutlass_grit_500k_20261121/tail_cutlass-gt.json`
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

One measured wall figure: 17,102 s (4.7506 h) is the last `elapsed_s` of
`train_log.jsonl`, A100, training only. CPU hours, queue time and the preflight
stage are not measured.

## Unverified

- Cross-platform shift +0.035 meV is taken from the ruling; the DeMol decision
  was not re-read.
- The paired bootstrap against s44 was not run (not needed for the fail).
- Scoring ran on an RTX 5060 (torch 2.7.1), training on A100.
- Whether the gap comes from the mechanism or from the untuned recipe is not
  separated; no claim is made.

## Not authorised

Retry, grid, a 100K companion, any adoption, an official-valid or test read.
