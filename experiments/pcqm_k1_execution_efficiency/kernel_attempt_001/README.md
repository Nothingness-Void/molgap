# Isolated kernel-trace diagnostic

Desktop NO_TRAIN continuation; prior TF32 permission improved short step time,
but the bundled profiler returned INVALID_DEVICE. Test the unchanged bundled
CUPTI12.8 and independently loaded official CUPTI12.9.79 in separate processes,
each with strict FP32 or matmul-only TF32. Never replace torch DLLs, drivers or
the project environment. Official SDK retrieval is infrastructure preparation.

Reuse the accepted train-only4096-row probe, first256 members, frozen initial
state, K1 family step/optimizer and precision/layout contexts. Three discarded
warmup updates then two traced updates, physical BS128, seed42, four CPU threads.
No scientific training, development/protected role, checkpoint or quality claim.
Concurrent-kernel activity is attribution, not a throughput benchmark. Unavailable
traces stay unavailable; missing timestamps/drop records prevent timing authority.
Raw kernel names alone do not establish tensor instruction counts or utilization.
The isolated collector supports only verified API28/Kernel9 ABI. If the bundled
library reports another API, record unsupported collector ABI, not device failure
or proof that a library upgrade caused a tracing fix.

Prospective before execution, idle-only GPU,300s total cooperative ceiling and
bounded child timeout. Failed cases retain error, observed work and native cost;
never silently retry. Closure is NO_TRAIN regardless of profiler availability.

Official SDK: https://developer.download.nvidia.com/compute/cuda/redist/cuda_cupti/windows-x86_64/cuda_cupti-windows-x86_64-12.9.79-archive.zip
Archive SHA256:c96018456dc4db6af6d69d6db0170a2ccc656ddde8c8ce6ee05682c2c5569daa.
See [CUPTI12.9 support](https://docs.nvidia.com/cupti/12.9.1/release-notes/release-notes.html)
and [activity semantics](https://docs.nvidia.com/cupti/12.9/api/group__CUPTI__ACTIVITY__API.html).
