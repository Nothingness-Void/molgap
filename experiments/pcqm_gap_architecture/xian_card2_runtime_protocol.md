# Xi'an Card2 runtime qualification

## Question

Can the SCNet Xi'an 16 GB Card2 allocation execute bounded PCQM Gap
architecture screens at useful throughput without weakening artifact or data-role
controls?

This is a platform qualification, not an architecture comparison. It cannot
change the frozen GraphState winner or authorize official PCQM roles.

## Boundary and budget

- Remote paths are restricted to `/work/home/acf9jvb3sm/`.
- At most 10 accelerator-hours may be consumed by environment repair and
  qualification.
- No local model execution is allowed.
- The immutable ETKDG geometry cache aggregate is
  `3e4206fd239942ab79f9c4978cd4334f6025ba14e8f1cd1df78c20060a0d1d22`.
- Official validation and test-dev remain unread.

## Runtime identity

- Device: one Xi'an `Z100SM` Card2, 16 GB.
- Vendor stack: PyTorch `1.10.0a0+git2040069-dtk2210`, HIP
  `5.2.22451-8f78d635`.
- Candidate: frozen GraphState9, source
  `9068ddb82e6bdf16b841570abbff023b90c07f07`, 3,665,809 parameters.
- Optimizer: AdamW, learning rate `1.6e-4`, weight decay `1e-6`; direct Gap,
  FP32, seed 42.

The platform lacks a compatible packaged PyG scatter extension. The isolated
environment therefore builds `torch-scatter 2.0.9` for the vendor HIP stack.
The scatter launch width is fixed at the device limit of 256 threads; all
source tarballs and the resulting wheel are hash-pinned.

## Gates

1. The runtime and model preflight must produce finite forward/backward values.
2. The transferred 100K/10K cache must pass independent no-model acceptance.
3. A 10,000-train-graph probe compares batch 48, 96, and 192 with the pinned
   T4 reference of `566.5467159302983` graphs/s.
4. The exact historical batch-48 contract must finish three full epochs with
   atomic progress, best model, last checkpoint, finite metrics, and telemetry.
5. Because the vendor scatter path is not bitwise deterministic, an unchanged
   repeat audits whether validation behavior is reproducible enough for paired
   ranking. No architecture search is released before coordinator review.

For future Xi'an screens, a different batch size is allowed only when both the
fresh comparator and candidate use that same Xi'an contract. Such results are
not arithmetically compared with historical batch-48 metrics.

