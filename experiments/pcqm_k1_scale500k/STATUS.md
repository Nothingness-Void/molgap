# Status

The K1 independent shadow audit passed and the paired 500K scientific contract
remains eligible. Kaggle2 cache versions 1 and 2 both stopped before graph
construction for infrastructure-only reasons: version 1 could not resolve the
extracted source tree; version 2 omitted RDKit from its isolated dependency
installation.

No GPU training was submitted and no scientific result exists. The exact
version-2 evidence is `terminal_report_cache_v2.md`. A further run must retain
the frozen split, architecture, seed 42, FP32, physical batch 128, optimizer,
schedule, 40 epochs, and decision gate.
