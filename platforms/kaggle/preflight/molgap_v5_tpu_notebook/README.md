# Kaggle TPU notebook allocation probe

This bounded probe distinguishes Kaggle script-kernel accelerator behavior from
notebook-kernel allocation. It mounts no data, performs no training, and writes
only `tpu_allocation_report.json`.

Submit it with the official Kaggle CLI and explicit `--accelerator TpuV5E8`.
Acceptance requires JAX to expose at least one TPU device. Metadata alone is not
accepted as evidence of accelerator allocation.
