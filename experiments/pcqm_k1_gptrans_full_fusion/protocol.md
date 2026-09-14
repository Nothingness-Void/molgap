# Full K1 and GPTrans-T comparison

## Question

Can the accepted K1 and GPTrans-T candidates retain their 500K-scale promise
when each is trained on the complete accepted PCQM4Mv2 official training role,
and does a separately calibrated prediction blend improve over either model?

## Frozen inputs and exposure

- Dataset root: `pcqm4mv2-fixed-v1`, accepted train-only topology cache.
- Exactly 3,378,606 official training rows; canonical manifest SHA-256
  `7d358a679d299bd5de6c5822e42dcef34b0e0b9fa9598c684709509c0b94021d`.
- Topology aggregate SHA-256:
  `be18a9964754c8428b5c62433136bd0bafb6c75016811c79e223aec30f3dbe98`.
- Both runs use the privately staged OGB 1.3.6 source archive with SHA-256
  `a18d4cacc6a35ad24938f52cfe197a255a5f64bb197f8d0f056c204467ec1e33`; the
  runtime manifest records this separately from installed distributions.
- No development, official validation, test-dev, or challenge-test rows are in
  this cache. The two base runs do not read those roles.
- Both candidates use seed 42, FP32, TF32 disabled, deterministic algorithms,
  one A100, physical batch 128, and exactly 20,000,000 sample presentations
  (156,250 optimizer steps). Incomplete final batches are dropped per global
  pass.
- K1 follows its frozen full-role contract unchanged. GPTrans-T retains its
  accepted 500K architecture and optimizer recipe; its four warmup epochs are
  represented as four equal exposure stages, followed by cosine decay over the
  same 20M presentation budget. GPTrans-T uses its accepted EMA decay 0.9999.
- Because optimizer, learning rate, weight decay, and EMA differ by candidate,
  this is a matched-data/exposure recipe comparison, not a single-variable
  architecture ablation.

## Execution and durability

Each model has a separate A100 preflight and train output. A preflight checks
the exact runtime, model parameter count and initialization hash, cache hashes,
full-batch forward/backward, memory headroom, and projected wall time. A failed
preflight blocks its training job. Each run writes atomic checkpoints every
500 optimizer steps, records source/cache/runtime identities, and can resume
only from a matching checkpoint. Both base-training jobs keep official
validation and test roles sealed; the later fusion study has its own explicitly
defined official-valid calibration/holdout access below.

## Fusion study

An accepted official-valid graph cache is available at
`/lustre/home/users/sm2/chou/molgap-pcqm-edge-state-full/rich_full/graphs`.
It contains 73,545 official-valid rows, uses the accepted OGB feature schema,
and matches the full-train source-row manifest. The user authorized a fusion
experiment on 2026-09-13. This study reads official validation once and spends
that role on a pre-registered calibration/holdout analysis; it is not an
untouched official score for later model selection.

- Sort by `source_idx`; calibrate on rows where `source_idx % 5 == 0` and do not
  fit on the other four fifths.
- Fit only one convex scalar coefficient for K1 versus GPTrans-T. Search the
  closed interval [0, 1] on a fixed 0.001 grid, minimizing calibration MAE.
- Compare both single models, fixed 50:50, and the calibrated blend on the
  four-fifths holdout. Also report fixed 50:50 on all official-valid rows; this
  fixed blend has no fitted parameter.
- Save prediction chunks atomically with input/model hashes. A separate CPU
  acceptance job verifies identity, alignment, metrics, and artifact hashes.
- Never read test-dev or challenge-test. Do not represent the calibrated
  official-valid result as a leaderboard/test result, or repeat tuning on this
  consumed validation role.

Mechanical artifact acceptance does not itself imply scientific advancement;
holdout deltas determine whether the combination is useful.
