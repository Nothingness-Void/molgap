# Loader failure-mode attribution

The accepted discriminator is cold startup versus actual collation, not a
comparison of model endpoints. Windows worker startup dominates this short
CPU-only4096-row probe; collation itself is millisecond-scale. The sender creates
a new loader each arm/epoch, so its persistent workers cannot persist across
those instances. This is a concrete avoidable path, not evidence that CPU work
explains all A100/T4 parity.

The previous rejection was CPU isolation: empty CUDA visibility did not make
`torch.cuda.is_available()` false in this Windows runtime. Actual visibility `-1`
removed that trigger without changing the loader's pin-memory configuration.
Both exact batch checks and owned-worker cleanup now pass.

Missing discriminator: native sender phase timing and a matched persistent-pool
contrast. Effective pinning and GPU-consumer overlap are absent. Frozen model
and scientific decisions remain unchanged. Next permitted work is bounded,
prospective TRAIN-only CPU reuse qualification, not another training run.
