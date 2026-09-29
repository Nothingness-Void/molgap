# T4x2 prospective planning decision — 2026-09-29

The P100 route closed `NO_TRAIN` because the requested Kaggle accelerator was
retired before submission. The scientific question remains unresolved and the
already accepted V4 reference remains reusable. Prepare one T4x2 candidate
prospective declaration on the same experiment branch without retraining a
reference.

The user requested a local-versus-remote speed comparison, not remote GPU
execution. This decision publishes a reviewable local plan and source package;
it does not submit, allocate, or authorize an immediate training run. A later
remote action must pass source, data, role, same-device latency, memory,
repeatability, and T4 native-cost gates. No protected evaluation role, second
seed, 500K, full-scale training, or automatic promotion is released.
