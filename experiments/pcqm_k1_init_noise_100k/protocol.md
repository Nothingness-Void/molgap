# K1 post-BN 100K initialization noise

Prospective desktop question, 2026-10-30. Two kernels from one study, each with a
same-run seed-42 reference and one seed-varied candidate. The user authorized both
(relayed by the coordinator). Owner: codex/exp/pcqm-k1-init-noise-100k.

## Question

How much does K1 post-BN 100K development MAE change with the initial weights
alone, and what does a 3-init mean prediction gain over a single run? No 100K K1
multi-seed result exists, so the 1-3 meV near misses (typed pair token 2.80 meV,
combined simplification 2.16 meV) cannot be read against a noise floor.

## Design

- Kernel A: kaggle2 (kaseichou), `s43/`, arms `ref_s42`, `init_s43`.
- Kernel B: kaggle3 (nvoid912), `s44/`, arms `ref_s42`, `init_s44`.
- `ref_s42` is the frozen post-BN preset (recipe, data declaration, init state
  `8ef6d4ba...`). `init_s43` and `init_s44` use `build_initial_state(43)` =
  `5b8b6b56...` and `build_initial_state(44)` = `7990bcf3...` (see each
  `initialization_provenance.json`). The seed-42 build reproduces the preset hash.
- Held fixed: recipe, data order (sampler seed 42), dropout RNG, no addons.
- The spec seed label stays 42 because the compiler permits only seed 42. The true
  torch seed is recorded in the provenance file, as in the 500K initialization
  precedent. The candidate recipe and data declaration are copies that change only
  the initialization sha, arm id, role and recipe sha.
- Measures init variance only. Full seed variance is at least this large.
- 100K train rows [0,100000), development [100000,150000), 40 passes per arm,
  FP32. Official validation, test_dev and test_challenge are never read.

## Pre-registered reading

Measurements: spread D = max pairwise |MAE difference| among the s42 mean, s43
and s44; rerun drift R = |ref_s42(A) - ref_s42(B)|; ensemble gain E = MAE of the
3-init mean prediction against the mean single-run MAE, from saved 50K dev predictions.

- D <= 1.0 meV: init noise is small. The 2-3 meV near misses deserve a 2-init confirmation.
- 1.0 < D < 2.5 meV: keep the 3 meV gate. Any confirmation needs 2 inits per arm.
- D >= 2.5 meV: single-seed 3 meV screens are underpowered. Past near misses become
  unresolved, not positive; future screens need >=2 inits per arm.
- E >= 3 meV: ensembling beats every module tried. It becomes the leading 500K/full
  candidate for the owner to decide.
- E < 1.5 meV: same-architecture ensembling has low value at 100K.
- R > 0.5 meV: hardware nondeterminism is material.
- If only one kernel finishes: report one |delta| as a single noise sample; no D threshold.

Falsifier: if D < 1 meV and E < 1.5 meV, the explanation that noise and variance
hide gains is wrong, and the limit is capacity, data or contract.

## Limits

- No retry, continuation, 500K, full training or adoption. No protected roles.
- Ensemble E is offline analysis of development predictions. Each arm saves
  `development_predictions.pt` with `source_idx`, `target_eV` and `prediction_eV`
  (`family_training_outputs.py`), so E is measurable once both kernels are accepted.
- Row bootstrap is not training stochasticity; D is the noise estimate.
- Literature context (3D auxiliary supervision, ensembling, long training, more
  data) is background only. 3D is excluded by the Pure-2D constraint, and 160
  epochs gained only about 1.5 meV in the norm-exposure study.
