# K1 100K init-noise terminal decision, 2026-10-30 JST

Two Kaggle kernels completed both 40-pass arms each: kaseichou kernel
`molgap-k1-init-noise-100k-s42` (run `pcqm-k1-init-noise-100k-s43`, arms ref_s42
and init_s43) and nvoid912 kernel of the same slug (run
`pcqm-k1-init-noise-100k-s44`, arms ref_s42 and init_s44). Both retrieved with
`--all-arms --strict` and mechanically verified. Scientific decision is separate
from strict evidence qualification; the staging reports own verification status.

Only the initial weights differ between arms (torch seed 42 frozen, 43, 44).
Recipe, train rows [0,100000), development rows [100000,150000), sampler seed 42
and Live development selection are shared. The spec seed label stays 42 for all
arms; the true init seed is in `s43/initialization_provenance.json` and
`s44/initialization_provenance.json`.

## Measurements (50000 internal-development molecules, rows and targets aligned)

| Arm | Selected development MAE, meV |
|---|---:|
| ref_s42 (kaggle2 kernel) | 141.2945 |
| ref_s42 (kaggle3 kernel) | 141.2945 |
| init_s43 | 139.6632 |
| init_s44 | 140.6475 |

- Rerun drift R = 0.0 meV. The two ref_s42 prediction tensors are bit-identical
  across accounts, so same-seed training on T4 is deterministic here and the
  second reference arm duplicated the first.
- Init spread D (max pairwise among ref, s43, s44) = 1.631 meV. This falls in the
  pre-registered band 1.0 < D < 2.5 meV: keep the 3 meV gate, and any
  confirmation of a 1-3 meV result needs at least 2 inits per arm.
- Ensemble gain E = 7.683 meV: the 3-init mean prediction scores 132.852 meV
  against a mean single-run 140.535 meV. The 2000-resample paired-row 95%
  interval (seed 42) is [7.499, 7.868] meV and measures row noise only. Pairwise
  residual correlation is 0.848 to 0.860. The pre-registered rule E >= 3 meV is met.
- Candidate minus reference: init_s43 -1.631 meV (row interval [-2.486, -0.711]),
  init_s44 -0.647 meV (row interval [-1.508, +0.279]). Neither reaches the frozen
  3 meV material gate. The intervals do not measure training variability.

## Reading under the pre-registered rules

- The falsifier (D < 1 meV and E < 1.5 meV) did not trigger. Init alone moves
  dev MAE by up to 1.6 meV, which is the same size as the 1-3 meV near misses in
  earlier single-seed screens. Those screens cannot be called real gains or
  noise without at least 2 inits per arm.
- Same-architecture 3-init averaging beats every single module tried at 100K.
  It costs three times the training. It is a lead candidate for the owner to
  consider for 500K, not a released scale-up. Earlier module gains measured at
  100K did not hold at 500K (for example PairToken +3.044 meV at 100K against
  -0.699 meV at 500K), so the 500K value of ensembling is not established.
- Limits: each member's checkpoint was selected on the same development cohort,
  so every MAE here is mildly optimistic, though the ensemble mean itself was
  not tuned. D rests on three points, one of which is the reference. Data order
  and dropout RNG were held fixed, so full seed variance is likely at least this
  large; this run does not measure it.

## Closure

- ref_s42 closes **CLOSED** in both runs. The second run's reference is a
  duplicate of the first and adds no evidence.
- init_s43 and init_s44 close **NEGATIVE_UNDER_CONTRACT**, retaining their
  numerical results. They fail the 3 meV gate against the reference. No
  adoption, 500K/full scale-up, continuation or retry is released.

## Roles and cost

Each arm's `data_open_provenance.json` is hash-bound in its output manifest. It
shows the training process opened the 3 selected train/development shard files
and no official_validation, test_dev or test_challenge file. It covers
Python-level file opens in the main process only, not native code, subprocesses
or network. Roles are drafted from that record in `roles_s43/` and `roles_s44/`.

Pair process wall: 6375 s (kaggle2) and 5776 s (kaggle3), each on two Tesla T4s.
The retained execution ledgers own the exact device-hours. Queue and
pre-entry allocation time are unmeasured.
