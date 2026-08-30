# PCQM Geometry Fusion Seed-42 GPU Compatibility Failure

This is a dated infrastructure-failure record, not a model-accuracy result.

## Outcome

- Kernel: `kaseichou/molgap-pcqm-geometry-fusion-s42`
- Failed version: 1
- Repair version: 2
- Completed candidates in version 1: 0

Version 1 reached the Pascal compatibility bootstrap on a P100 with compute
capability `(6, 0)`. The bootstrap installed `torch==2.5.1` from the CUDA 12.4
wheel index, but that build did not contain `sm_60`, so the restart guard
raised:

```text
RuntimeError: PyTorch remains incompatible with compute capability (6, 0)
```

No candidate, epoch, geometry-cache record, or accuracy result was produced.
The failure payload and log were retained under
`platforms/_records/kaggle/training/pcqm_gap100k_geometry_bottom_fusion_seed42_v1`.

## Infrastructure-only repair

The runner was changed to detect `sm_60` explicitly and install the
P100-compatible `torch==2.7.1` CUDA 12.6 build together with
`nvidia-cusparselt-cu12==0.6.3`. The architecture, source rows, split, target,
seed, optimizer, schedule, precision, parameter budget, candidate order, and
sealed-role flags were unchanged. The exact package and evidence hashes are in
[`gpu_repair_manifest.json`](gpu_repair_manifest.json).

The dedicated static contract test passed, and version 2 subsequently reached
terminal `COMPLETE`. Its scientific disposition is separate in
[`decision.md`](decision.md).
