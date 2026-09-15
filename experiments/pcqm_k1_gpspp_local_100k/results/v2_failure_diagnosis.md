# Version 2 infrastructure failure — 2026-09-15

Kaggle2 kernel `kaseichou/molgap-k1-gpspp-local-s42` version 2 verified the
immutable source, installed the frozen runtime and launched both isolated T4
workers. Both stopped during train-only architecture preflight before epoch 0
with `NameError: name 'MIXER_LAYERS' is not defined` in
`k1_gpspp_local.check_mechanism`.

The constant had been imported inside `make_encoder`, where Python scoped it
locally, while the independent preflight checker referenced it at module
scope. The repair moves the same unchanged tuple import to module scope. It
does not change model tensors, equations, initialization, data, seed, batch,
precision, optimizer, schedule, exposure or role access. No development or
sealed-role result was produced, so version 2 is an infrastructure failure and
not a scientific observation. A contract-identical version 3 retry is allowed.
