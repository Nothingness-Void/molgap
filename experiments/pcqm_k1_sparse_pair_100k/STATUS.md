# Status

The frozen model source is commit `fb35e8b` with source-archive SHA-256
`c619a537ffe725eee9a42798e217a9b8dacd4d69205a1825608fab484a4f02b3`.

- Kunshan profile job `122512473` was cancelled after the user explicitly
  selected Kaggle for training. Superseded pending jobs `122511355` and
  `122511965` were cancelled before allocation after their incomplete source
  dependency closure was diagnosed from the Kaggle preflight.
- Kaggle kernel `nothingnessvoid/molgap-k1-sparse-pair-profile-b6dd3b8`
  version 3 completed and passed on a verified T4x2 allocation with one
  isolated T4: `1068.9733 graphs/s`, `1.0391 h` projected training time, and
  `95.84%` memory reserve. Versions 1-2 failed before model construction while
  establishing complete source/cache compatibility; their logs are preserved
  under `platforms/_records/kaggle/training/`.

- Kaggle training kernel
  `nothingnessvoid/molgap-k1-sparse-pair-train-s42-fb35e8b` version 1 was
  submitted with the accepted profile, immutable source bundle, fixed 100K
  data, FP32/no TF32, physical BS128, and the frozen 40-epoch schedule.

No scientific training result exists yet. The second-platform profile gate was
not completed; Kaggle release was an explicit operational selection and is
recorded as such rather than reported as a dual-profile comparison result.
