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
only from a matching checkpoint. Official validation and test roles remain
sealed.

## Fusion gate

The accepted full cache contains no independent development/calibration rows.
The base models train on every official training row, so fitting a blend weight
on their in-sample predictions would be leakage and is not accepted evidence.
The user authorized the two full training runs on 2026-09-13, but did not
specify a calibration role. A learned blend therefore remains gated on a
separately accepted, source-aligned calibration cache that neither base model
trains on. Official-validation graphs can support a calibration/holdout split
only after an explicit role decision; test-dev and challenge-test remain
forbidden. A fixed 50:50 prediction average needs no training labels and can be
evaluated once on the authorized final evaluation role.
