# K1 post-BN 100K initialization noise (server, seeds 45 and 46)

Prospective server measurement, designed 2026-11-03. It ports desktop's
`pcqm_k1_init_noise_100k` construction. This is a measurement: it has no
pass/fail gate, and no automatic verdict is to be read from it.

## Design

- One kaggle3 (`nvoid912`) T4x2 session, one arm per device: `init_s45` (same-run
  reference, needed only to run the pair) and `init_s46`.
- Both arms use the frozen K1 post-BN recipe and data declaration
  (`experiments/presets/k1-post-bn-100k-v1.json`). Only the initial state differs:
  torch seed 45 gives `1bae04de...` and seed 46 gives `db5f7aaa...`, from
  `make_encoder("neural_atom_k1")`. [build_initial_states.py](build_initial_states.py)
  writes the states, the per-arm recipe and data declaration, and each arm's
  `initialization_provenance.json`. Before writing anything, it reproduces the
  preset's seed-42 state and desktop's seed-43/44 states and recipe/data bytes.
- The spec seed label stays 42 because the compiler permits only seed 42. Data
  order (sampler seed 42) and dropout RNG are therefore fixed: this measures
  initialization variance only, and full seed variance is likely larger.
- 100K train rows [0,100000), development [100000,150000), 40 passes, FP32,
  physical batch 128. Official validation, test_dev and test_challenge are never read.

## What is reported

The selected development MAE (meV) per initial state: seed 42 141.2945, seed 43
139.6632, seed 44 140.6475 (desktop, accepted), plus seeds 45 and 46 from this run.
The server's seed-42 value is bit-identical to desktop's.

- Range: max − min over n=5.
- Sample SD s (n−1 denominator), with a chi-square 95% CI for df=4:
  [s·sqrt(4/11.1433), s·sqrt(4/0.48442)] = [0.599·s, 2.874·s].
- Paired-delta SD = sqrt(2)·s: the init-only noise of a candidate-minus-reference
  difference between two single-seed runs.
- Ratio of the 3 meV material-gain gate to the paired-delta SD.

The same development cohort is used for checkpoint selection and scoring, so the
selected MAE is optimistic. The within-run row bootstrap is not training noise.

## Limits

No successor, extra seeds, continuation, 500K, full training or adoption. No
protected roles. The later closure uses desktop's `closure_<run>.json` shape (per
arm `outcome` and `decision_ref`, plus `role_audit`). The reference closes
`CLOSED`. The `init_s46` outcome is a human decision; desktop's seed-varied arms
closed `NEGATIVE_UNDER_CONTRACT`.
