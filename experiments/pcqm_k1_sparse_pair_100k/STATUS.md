# Status

The frozen model source is commit `cb5e80c` with source-archive SHA-256
`9f155ede617b13a45de2991c9e0e19ad50790140f7f9f1cfe73e115f5c1768ba`.

- Kunshan profile job `122511965` is queued from the accepted fixed 100K
  cache. Superseded pending job `122511355` was cancelled before allocation
  because its source bundle was missing `constants.py`.
- Kaggle kernel `nothingnessvoid/molgap-k1-sparse-pair-profile-b6dd3b8`
  version 2 is running on a verified T4x2 allocation with one isolated T4.
  Version 1 failed before model construction for the same missing source-file
  dependency; its log is preserved under `platforms/_records/kaggle/training/`.

No scientific training result exists yet. Training remains blocked until both
profiles pass and `accept_profiles.py` selects the faster platform.
