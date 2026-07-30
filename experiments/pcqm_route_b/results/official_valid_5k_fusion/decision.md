# PCQM Route B Fixed Official-Validation Decision

## Decision

On 2026-07-29, the development-selected `augmented_schnet` bounded fusion
passed the frozen Track B acceptance gate. The three-seed equal ensemble reached
`0.112011 eV` Gap MAE on 4,981 valid molecules from the fixed official-valid
5K subset.

This result was accepted as a PCQM Gap-only specialist. It was not a leaderboard
submission, did not use the official test or sealed 20K, and did not change the
production registry.

## Frozen Protocol

- Encoder and fusion selection used scaffold development only.
- The selected identity was fixed before official-valid labels were read.
- Encoders were tuned GPS9-192, tuned GPS11-160, fixed primary SchNet-176/160/6,
  and fixed augmented SchNet-176/160/6.
- The selected identity was the augmented SchNet prediction plus a bounded
  `+-0.10 eV` correction using all four frozen embeddings.
- Seeds 42, 43, and 44 were evaluated individually and by a fixed equal
  prediction ensemble.
- Graphs used the training-time `ETKDGv3 + MMFF200` contract. Both conformers
  were built to reproduce the accepted-row filter; inference used the primary
  view for both SchNet branches.

## Results

| Model | Gap MAE (eV) |
|---|---:|
| GPS9-192 | 0.181596 |
| GPS11-160 | 0.181966 |
| Primary SchNet | 0.126200 |
| Augmented SchNet | 0.122450 |
| Fusion seed 42 | 0.112118 |
| Fusion seed 43 | 0.112120 |
| Fusion seed 44 | 0.111985 |
| Fusion equal-seed ensemble | **0.112011** |
| GINE v7 fixed-valid reference | 0.184618 |

The equal-seed Fusion improved over the augmented SchNet identity by
`0.010438 eV` (`8.52%`) and over the GINE v7 reference by `0.072607 eV`
(`39.33%`).

## Integrity

- Source rows: 5,000.
- Accepted aligned rows: 4,981.
- Failed conformer rows: 19.
- Labels SHA256:
  `38025d214fba7995086ac0c24e54fb276d87ecaab7ebe97b10922081f0596877`.
- Development selection SHA256:
  `b441e166a67d233b788b40cdd5235e3a16b2b16d9aa4b85bdd95cd94bd37e455`.
- Graph manifest SHA256:
  `f81f1ff90345a96aed6158a321939d4be957fb9ab5da5e6c8d4fc774ee3062c7`.
- Deterministic cache replay produced byte-identical metrics:
  SHA256 `e231ed1eef22058374b339706c9a93a070beb7106ea304c8fcb7d0feb0b44ee7`.

Machine-readable metrics and row predictions are `metrics.json` and
`predictions.csv` in this directory.
