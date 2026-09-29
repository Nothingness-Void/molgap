# Kaggle1 paired attempt 1 stopped at GPU preflight — 2026-09-29

Kaggle1 created the private kernel under its observed canonical reference
`nothingnessvoid/molgap-gptrans-init-pair-100k-s42` (the first-create slug
differs from the locally requested longer ID). The Kaggle status reached
`KernelWorkerStatus.ERROR`. The downloaded `attempts/kaggle1_pair_v1/kernel.log`
and `submission_binding.json` bind this attempt to source commit
`e0f8493539e4d70deb96ef378b133ee39354b78a`, source package identity
`a811a0ba76f2df0f64a00cb10d1035f9fdf80017c7c51d9f8620e386e5bc08f4`,
and Spec identity `660f94cb5487828069fd529d20febd3bda07df5463e5b351f37005e8308b415d`.

The reference arm's single-T4 V4 preflight completed and accepted its runtime
certificate (`attempts/kaggle1_pair_v1/reference_preflight.json`). Candidate
preflight stopped before its own runtime certificate: the model's complete
initial state hashed to
`9af1db8997c8ac28dc085ee113fef23c639f700f45d915e40c4339ef2e181c6a`
on Kaggle's Torch 2.10.0+cu128 environment, whereas the prospective Spec
pinned the local Torch 2.7.1+cu128 result
`8e29d413636f2187a66fdb53d873a5990e57875f87a186a19b8dcf60dfb424b9`.
The original frozen V4 artifact and reference model identity passed their
checks. Different CPU normal-initialization streams across the two runtimes
are compatible with this observation; the exact library-level cause was not
isolated. No training epoch, candidate MAE, paired comparison, or protected
evaluation role was produced. Native T4 device time and queue time were not
reported and remain unknown, not zero.

Both trajectories close `INCONCLUSIVE` for this infrastructure mismatch. The
missing discriminator is a candidate preflight under a single frozen T4
initial-state hash. A corrected Spec may bind the actual observed T4 hash and
retry once with a new prospective source/config identity. This does not change
the 15-table mechanism, 100K decision gate, or reference history; it does not
constitute a scientific negative or release scale-up.
