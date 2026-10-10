# CPU snapshot decision

2026-10-10; Desktop NO_TRAIN. Source3d5f7b2d; [result](results/result.json)
and [protocol](protocol.md) own the measured scope. Worker wall28.878s,
process CPU64.188s; accelerator and queue not applicable. Preparation, tests and
closure costs excluded. All six exactness/scope checks passed; 107 focused tests
passed separately with one PyG deprecation warning.

Only a retained44MB scratch checkpoint was decoded on CPU. No model, optimizer,
data loader, molecular labels, inference or gradients were created/consumed.
Telemetry values are synthetic IO fixtures, not historical learning records.
The hash-pinned sender copier was AST-isolated, not a remote runner import.

| Synthetic rows per arm | Original copy (s) | Candidate copy (s) | Original copy + publish/hash (s) | Candidate copy + publish/hash (s) |
|---|---:|---:|---:|---:|
|0|0.005967|0.005563|0.084229|0.082565|
|128|0.006973|0.005693|0.087119|0.102105|
|8192|0.060926|0.020576|0.252854|0.214214|
|32768|0.229311|0.056235|0.754504|0.594260|
|65536|0.443833|0.104448|1.379635|1.040663|

Three alternating-order repetitions per size; values are medians. The largest
fixture reduces copy time76.47%, but complete measured publication only24.57%.
The128-row publication is slower: filesystem variability remains visible and
the small sample does not qualify universal IO improvement.

Nominate `checkpoint_to_cpu` as an opt-in helper only. Logical values, tensor
clones, independent storage (synthetic unit tests), CPU AdamW step tensors,
roundtrip state, input and RNG are preserved. CUDA stayed uninitialized.
No frozen sender/default/active job changed. No checkpoint frequency, format or
log retention was reduced. No real end-to-end training saving, Drive/Kaggle
throughput, crash durability, native T4/A100 qualification or retrained MAE claim.

Before publication, custody rejected a Git/checkout CRLF mismatch. Exact bytes
were retained by the scoped Git attribute change; no arithmetic changed and
the diagnostic began only after the prospective plan was successfully frozen.
