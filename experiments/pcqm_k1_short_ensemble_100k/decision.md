# K1 100K short-ensemble: terminal decision, 2026-11-01

One kaggle2 (kaseichou) kernel `molgap-k1-short-ensemble-100k-s42`, kernel ID 141973073,
version 1, run `pcqm-k1-short-ensemble-100k-k2-r2` attempt v1, finished both 20-pass arms
`short_s43` and `short_s44`. Retrieved with `--all-arms --strict`. Only the initial weights
differ between arms (torch seed 43 and 44); the recipe is the retained post-BN 100K recipe
with `total_units` 20. Protected roles official_validation, test_dev and test_challenge are
`untouched` in both role drafts (derived from `data_open_provenance.json`).

## Measurements (50000 internal-development molecules)

Source: `analyze --op summary` on the two `development_predictions.pt` files
(`source_idx_sha256` 3270b300..., same rows as the retained 40-pass files).

| Model | Dev MAE, meV |
|---|---:|
| short_s43, 20 passes | 150.875 |
| short_s44, 20 passes | 148.942 |
| mean of the two 20-pass singles | 149.909 |
| 2-member ensemble of the two 20-pass models | 145.075 |
| retained 40-pass singles (init-noise study): mean S40 / best B40 | 140.535 / 139.663 |

- Ensemble gain over the 20-pass singles: 4.834 meV. Init spread D = 1.933 meV.
- Cost of halving the schedule: the 20-pass single mean is 9.374 meV above S40.
- Equal-compute ensemble (2 x 20 passes) against one 40-pass model: 145.075 against 140.535,
  4.540 meV worse than S40 and 5.412 meV worse than B40.
- Row-bootstrap 95% interval for short_s44 minus short_s43: [-2.773, -1.128] meV. It measures
  row noise only.

## Reading under the pre-registered rules

- POSITIVE (ensemble <= 137.535 meV): not met.
- NEGATIVE (ensemble >= 139.663 meV): met (145.075).
- Falsifier (20-pass single mean more than 8.0 meV above S40): triggered (9.374 meV). The
  shorter schedule loses more than the ensemble recovers, as the rule anticipated.

Both arms close **NEGATIVE_UNDER_CONTRACT**. They are two members of one offline ensemble,
not candidate recipes. No retry, continuation, 500K, full training, distillation or adoption
is released. The accepted 100K recipe is unchanged.

## Terminal attribution

- Measured: halving the cosine schedule costs one K1 model about 9.4 meV; a second
  independent 20-pass member recovers 4.8 meV. In the 40-pass study the same step (N=1 to
  N=2) recovered 5.70 meV, so the ensemble gain per added member is similar at both lengths
  (4.83 against 5.70) while the single-model cost of 20 passes is not.
- Not established: why the 20-pass models are worse (insufficient training exposure, schedule
  shape or something else). One schedule length, one pair of inits and no 20-pass trace
  comparator mean the data cannot separate these; the verdict is `insufficient_evidence`
  for any mechanism. Missing discriminator: a third schedule length, or members at 30 passes.

## Limits

- The 40-pass comparators ran in other kernels and sessions (kaggle2 and kaggle3 init-noise
  study). The retained `ref_s42` predictions were bit-identical across the two accounts, which
  supports a cross-kernel comparison on T4, but this run did not retrain a 40-pass arm.
- Two members, one data order and one dropout RNG; full seed variance is likely larger than D.
- Each member's checkpoint was selected on the same development cohort, so every MAE is mildly
  optimistic. The ensemble mean was not tuned.
- Ensemble numbers are offline analysis of saved predictions, not a trained model, and the
  ensemble needs two forward passes at inference.
- Row bootstrap is not training variability.
