# Fixed-data GPU version 2 infrastructure failure

Kaggle2 kernel `kaseichou/molgap-pcqm-k1-scale500k-s42` version 2 terminated
before loading any graph or training any model. The runtime exposed one GPU,
while the frozen paired task requires two isolated T4 devices. The entry point
failed closed at its device-count guard after about 11 seconds.

The source, fixed dataset, architecture, seed, FP32 precision, physical batch
128, optimizer, schedule, epochs, role boundaries, and sealed-data flags were
not exercised or changed. The failure was caused by submission through the
legacy CLI path without the explicit accelerator override. Version 3 therefore
uses the unchanged package and source commit with explicit
`--accelerator NvidiaTeslaT4`; this is an infrastructure retry, not a new
scientific attempt. The raw terminal log is retained at
`platforms/_records/kaggle/training/pcqm_k1_scale500k_fixed_s42_v2/`.
