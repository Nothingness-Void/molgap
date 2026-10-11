# Explicit infrastructure retry

2026-10-11 JST. User explicitly requested repair and resubmission after
[attempt001 failed before training](../terminal_acceptance/decision.md).
This is the same scientific question and owning branch, not a model successor.

The [original protocol](../protocol.md) remains the scientific/resource authority:
fixed100K plus internal50K development, seed42 full initial tensors, original
single-forward versus fused AdamW/CPU-layout, FP32/BS128 and40epochs per arm.
Quality/cost gates are unchanged. No protected roles, new optimizer/model/data
recipe, automatic retry, continuation or adoption.

Only shared initial transport reading and the local frozen-package release check
are repaired. Both generic inspection and the actual family loader must validate
the same uploaded wrapper; no tensor regeneration or weakening of hash checks.
The source commit/archive, Spec, per-arm prospective records and job are newly
bound for this attempt, preserving the original attempt and frozen source.
Private source v2 and kernel v2 are distinct from v1; old source is not versioned
or replaced. All-arm native T4 qualification remains mandatory before training.

[User release](user_release.json) owns retry authority; [parent handoff](../REMOTE_HANDOFF.md)
owns reconciliation navigation. Platform API receipts belong in this attempt's
submission directory. Missing native qualification/result remains pending.
