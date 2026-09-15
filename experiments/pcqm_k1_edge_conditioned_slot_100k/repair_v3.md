# Version-3 infrastructure repair

Versions 1 and 2 requested a P100 but Kaggle exposed a Tesla T4. Both launchers
failed before model import or training, so neither is a scientific attempt.

Version 3 removes accelerator-model identity from the scientific launcher
contract. It exposes only the first assigned GPU through
`CUDA_VISIBLE_DEVICES`, verifies that exactly one CUDA device is visible, and
verifies that the installed PyTorch contains the device's compute capability.
The compatibility wheel is installed only when that check fails. The launcher
therefore accepts an assigned P100, T4, or another supported CUDA accelerator,
while the existing runner records the exact device and optimizer-inclusive
runtime certificate.

The architecture, source payload, fixed PCQM-100K data identity, row order,
seed 42, direct Gap target, strict FP32/no-TF32 mode, physical batch 128,
optimizer, scheduler, 40 epochs, sample exposure, selection rule, role access,
checkpointing, and promotion gate are unchanged. This repair changes only
resource binding and dependency compatibility.
