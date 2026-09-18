# GPTrans Pair-Update-Norm 500K Profile

This is a bounded, training-free performance qualification for the accepted
`pair_update_norm` mechanism on Kaggle3 T4 hardware. It mounts the immutable
500K V5 cache but reads only the first accepted 50K training shard.

The run keeps the scientific execution invariants that affect comparability:
FP32, no TF32, deterministic algorithms, one visible GPU, physical batch 128,
and the unfused/`foreach=False` AdamW optimizer. It changes only DataLoader
settings and measures optimizer-inclusive throughput, loader wait, host-to-
device transfer, CUDA compute, and peak memory.

The kernel is `nvoid912/molgap-gptrans-pair-update-500k-profile-v1`. Its result
can choose a runtime configuration for a later separately authorized 500K
bridge; it is not model-quality evidence and does not read development or any
official evaluation role.

