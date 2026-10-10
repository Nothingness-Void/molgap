# T4 paired500K native-cost control

2026-10-10 Asia/Tokyo. User authorizes Kaggle1 API submission, maximum nine
hours total runtime, with reported45 hours available. No browser submission,
purchase, successor, continuation or automatic A100 cancellation is authorized.
This is an execution control for the same open single-forward/EMA500K question,
not a repeat of a closed scientific experiment.

Reuse the [parent scientific protocol](../protocol.md): identical fixed500K
training / internal50K development, seed42 full initial-state bytes, FP32 with
TF32 disabled, physicalBS128, single forward, unfused AdamW, cosine60 horizon,
EMA0.999 and training-only clean BN calibration. No schedule compression,
protected roles, teacher, geometry input or independent-holdout claim.

Only platform/hardware and authorized allocation ceiling change. Freeze a new
source snapshot adding backward-compatible T4 qualification; the running Colab
source archive/config and its original four-hour deadline remain untouched.
Both arms execute serially on CUDA-visible device0, alternating complete epochs.
Kaggle may allocate T4x2; record both the physical allocation and one-device
usage, never label unused physical allocation free or infer provider billing.

The cheapest discriminator is the existing per-step trace and complete-epoch
wall intervals. Compare the same arm/optimizer coordinates and recipe; separate
setup, step execution, loader/checkpoint/evaluation overhead and native device
units. Early extrapolation is provisional, not a measured full60-epoch cost.
Cross-platform/runtime/CPU differences prevent attributing every timing change
solely to the GPU. No cross-hardware STRICT_CAUSAL scientific comparison.

All-arm real-GPU qualification must pass before formal training. Freeze source,
initialization, contract, input/cache identities and prospective RML before
submission. The platform retains kernel outputs containing source/config,
qualification, atomic last/best state, trace and terminal/hash manifests. Parent
watchdog bounds setup/worker/publication within32400 seconds; worker reserves
120 seconds. Budget stop is STOP_FOR_COST, not a complete endpoint. A worker
failure is infrastructure until qualified evidence proves otherwise. Reconcile
the exact returned API job/version; no speculative retry on an unknown state.

Only the user decides whether a measured T4 result warrants stopping A100.
Do not stop either job just because the other job has a lower partial MAE.
