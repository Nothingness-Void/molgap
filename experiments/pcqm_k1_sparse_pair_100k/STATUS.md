# Status

The frozen model source is commit `fb35e8b` with source-archive SHA-256
`c619a537ffe725eee9a42798e217a9b8dacd4d69205a1825608fab484a4f02b3`.

- Kunshan profile job `122512473` is queued from the accepted fixed 100K
  cache. Superseded pending jobs `122511355` and `122511965` were cancelled
  before allocation after their incomplete source dependency closure was
  diagnosed from the Kaggle preflight.
- Kaggle kernel `nothingnessvoid/molgap-k1-sparse-pair-profile-b6dd3b8`
  version 3 completed and passed on a verified T4x2 allocation with one
  isolated T4: `1068.9733 graphs/s`, `1.0391 h` projected training time, and
  `95.84%` memory reserve. Versions 1-2 failed before model construction while
  establishing complete source/cache compatibility; their logs are preserved
  under `platforms/_records/kaggle/training/`.

No scientific training result exists yet. Training remains blocked until both
profiles pass and `accept_profiles.py` selects the faster platform.
