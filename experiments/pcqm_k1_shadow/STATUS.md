# Status

The repaired label-sealed 10K shadow cache passed independent mechanical
acceptance. Its aggregate SHA-256 is
`4a9e4361247f497f5cd9911a69e69eddb1e0fecea6e33f318ada48aa594f6a44`;
all 10,000 graphs contain only OGB atom/bond features, RWSE16, and immutable row
identity. No target label or model inference entered cache construction.

Audit version 1 failed before model construction or label access because
`hasattr` is not a valid PyG store-membership test. The unchanged-contract fix
uses `"y" in graph`; its evidence is `terminal_report_audit_v1.md`.

The frozen one-time audit is now `RUNNING` as Kaggle2 kernel
`kaseichou/molgap-pcqm-k1-shadow-audit` version 2. It loads only the exact
accepted Full-GPS and K1 checkpoints, saves both prediction vectors before its
sole shadow-label access, and performs no optimization. Passing releases only
the already prepared 500K bridge to desktop; failure closes K1.
