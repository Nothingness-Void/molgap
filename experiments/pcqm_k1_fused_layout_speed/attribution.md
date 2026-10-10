# Execution attribution and exclusions

The local real-stream test confirms a persistent benefit of the joint
fused-AdamW/CPU-layout runtime over the explicit unfused control. It does not
independently identify each contribution. Both model factories, loss, precision,
initial tensors, sampler, exposure and cosine40 prefix are shared.

Observed shared loader wait is13.160s of1478.120s total two-arm wall (~0.89%).
Checkpoint publication is17.567s (~1.19%). Loader wait is not all worker CPU
time: asynchronous worker work may overlap GPU execution. Kernel phases include
host launches and synchronization, not exclusive accelerator-busy time.
The evidence does not support claiming that CPU graph loading explains most
of the A100/T4 parity. The earlier accepted A100 in-memory phase profile also
found little loader wait. Native-platform/end-to-end attribution remains open.

Limits: one local realization, background GUI/emulator activity retained, paired
alternating windows rather than independently executed standalone runs, explicit
foreach=False control rather than historical100K auto-foreach default, fused
rounding not bitwise identical, no development predictions and no accuracy test.
Resume state was retained, but this adapter's next-step replay was not certified.
Do not promote a model, infer train-fit/underfit, or extrapolate GPU savings from
this result. No additional run is released by this attribution.

Preparation receipts: first RML publication was blocked by a missing policy
source digest before any plan was published. After that repair, both original
prospective records were retained. Python3.10 rejected extractall(filter=...),
and the first bundle omitted the eager constants import. These were packaging
failures before model execution; the corrected source2 inventory contains the
same committed executable source plus its missing tracked import dependency.
The first archive and failed import console remain in ignored local staging.
The frozen training source was not edited during the run. No remote job changed.
