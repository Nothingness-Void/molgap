# S1-P0 and N4: chemistry in frozen PairNorm 500K errors

Local desktop Track B diagnostic authorized 2026-11-22. No backbone training,
remote compute, full-scale arm, tail, official validation/test or model adoption.
Fit small CPU postprocessors; this is NO_TRAIN only for the backbone.

## Evidence and question

The two Round 5 filings beside the supplied `final_ruling.md` disagree about
deployment, not the existence of signal in K1 development residuals. The
BN-calibrated K1 scaffold-split chemistry gain over recalibration was reported
as 4.12 meV. On two different K1 states, fitting in-sample residuals gave only
0.14-0.32 meV beyond recalibration. Neither number is a GPT-T measurement.
`experiments/pcqm_pairnorm_prediction_router_500k/decision.md` tests a blend,
not a descriptor correction. The B3LYP legacy head_posthoc residual calibrator
showed only 0.29 meV external Gap improvement on another task. The frozen
GPTrans readout diagnostic fitted on training rows but used hidden states.
N4 extends the local K1 100K error-localization audit to two GPT-T 500K inits.

Use strictly retrieved `pcqm-gptrans-pairnorm-init-noise-500k-s43/v1`,
`init_s43` and `init_s44`. The parent decision records 101.0866 and 101.3195
meV on the same selection-used 50K. Only initialization varied. No strict
new-recipe comparison, fresh seed-variance estimate or extrapolation to tail.

## Frozen design

- Dataset: pinned official PCQM4Mv2 archive, index and 2D SMILES on train
  [400000,500000) and dev [500000,550000) only. Skip earlier raw lines without
  parsing them; stop before row 550000. Check indices and cross-check the raw
  Gap labels against retained targets for all dev rows and selected train rows.
  Approved topology shards 8/9 (train) and 10 (dev replay) only. Do not invoke
  a role scorer CLI. Reuse its **model factory and prediction function only**.
- Nonchiral canonical Murcko scaffold; all acyclics have the same empty key.
  Full MD5 integer modulo 2: even fit, odd score. Invalid molecules share
  one explicit key and have missing chemistry, never dropped. No scaffold
  may cross fit/score for either fit source. On each side use the smaller of
  the two eligible fit counts; sample without replacement from the larger
  using NumPy default_rng(20261122), sort by index. The two inits use identical
  fit memberships. Score every odd dev row once. Save the actual row IDs.
- Dev fit: use the selected base's stored predictions. Train fit: replay the
  same selected checkpoint (not last/live) with its frozen transform; verify
  first 128 dev predictions to max absolute error <= 0.0001 eV before using
  local FP32 predictions. Stop this column if identity/replay is not valid.
  Original torch/runtime differs; this is numerical inference validation, not
  training replay. Replay failure cannot release another checkpoint or role.
- Features: all 217 installed RDKit 2D descriptors, six counts (conjugated
  bonds, atoms incident to them, aromatic atoms, double bonds, triple bonds,
  radical electrons), radius-2 1024-bin Morgan counts and the base prediction;
  no conformers, target-derived input, scaler fit on score, or feature search.
  Descriptor failures are NaN. Exact names and library versions are retained.
  The original court scripts are not present here; this is a new prospective
  test of the claim, not a byte-for-byte reproduction.
- Model: predict residual y - yhat with HistGradientBoostingRegressor,
  absolute error, 500 iterations, 63 leaves, learning rate 0.1, seed 0,
  no early stopping. Fit separately for each init and source. Do not tune.
- Yhat-only control: choose among identity, median shift, affine L1,
  isotonic and the same one-feature residual GBM using **fit rows only**.
  One fixed internal scaffold split (MD5 of `control:` + scaffold modulo 5,
  zero validates); choose lowest validation MAE, break ties in listed order,
  refit it on the entire fit partition. No use of score labels until outputs
  for both models and both fit sources are frozen. Report raw MAE, control,
  corrected MAE, and correction minus control in meV with a paired 2000-row
  bootstrap 95% interval (seed 20261122; conditional on models and split).
- S1 primary = dev-fit correction minus yhat-only. Both inits <= -2.0 meV:
  diagnostic PASS. Either > -1.0: KILL. Otherwise INCONCLUSIVE. Train-fit is
  a separate deployment analogue, **not** a second way to pass. Use identical
  scoring IDs in both. A positive is not an H1 or deployment release.
- N4: descriptive raw errors on all 50K dev rows, not a second selection
  opportunity. Predeclared overlapping strata: radical, no pi bond, ring
  count 0/1/2/3+, CHNO-only, halogen, S/P, heavy atoms <=10/11-15/16+,
  and repeated canonical isomeric SMILES in this dev cohort. Invalid listed
  separately. For each: count, each init's MAE, fraction of total absolute
  error and ratio to whole-cohort MAE. Co-miss = same signed error and
  both absolute errors >=150 meV; report its rate and enrichment. Flag only
  strata with >=10% of absolute error and >=1.5x global MAE in **both**
  inits, and >=1.5x global co-miss rate. Not evidence for a causal remedy.
- Budget: one local execution; estimated 1-3 CPU-hours, <=0.25 RTX 5060
  wall-hours for forward-only checks/predictions. CPU feature work and model
  fitting may take longer than the estimate. No A100. Keep local row outputs
  outside Git and aggregate results plus attribution in the owner branch.
  No retry on a scientific miss; an infrastructure interruption may resume
  completed identical stages without re-selecting or rescoring.
