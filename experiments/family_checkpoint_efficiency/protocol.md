# Checkpoint Diagnostic Contract

Question: does removing per-tensor synchronization and scalar-check overhead
reduce full checkpoint publication time without changing saved state?

Prior evidence: the accepted input-pipeline diagnostic found approximately
0.7ms loader wait versus53ms K1 update span on RTX5060; bucket sampling did not
produce a stable speed benefit. The shared session still checked every nested
accelerator tensor with a separate Python boolean, unlike its model-only checker.
Source review is a hypothesis, not measured end-to-end training attribution.

## Frozen Scope

Construct fresh K1 and GPTrans models using existing factories, never retained
trained weights. Simulate AdamW/EMA/cursor tensors with the actual model schema.
Do not construct graphs, read molecule caches or consume any dataset role.
No model forward, backward, optimizer update, MAE or production modification.

Compare the frozen original nested/model finite checkers with the shared batched
checker. Preserve checks of every model, optimizer, EMA, RNG and cursor tensor,
all host floats and first failing path. No validation cache or skipped guards.
Group CPU scalars by dtype, never narrow/convert their values.

Alternate original/batched order over five repeats after one warmup. Use cursor
lengths0,512,2048,3905. Measure finite checking separately; measure finite checking
plus atomic serialization/publication at512 and3905. Both arms use the same
payload and save path. Require identical SHA256 of both serialized checkpoints,
identical retained model/optimizer/EMA/cursor values and unchanged RNG state.
Time CPU and RTX5060 separately with synchronization outside timed GPU intervals.
Keep actual result ranges; a local SSD/synthetic checkpoint is not T4/A100 or
full500K training speed. Warm disk-cache writes are not a durability stress test.
Publication timing covers the session/writer finite checks and atomic Torch IO,
not graph loading, cursor construction, source/role admission or trace reporting.

Total native worker ceiling1200s; optional GPU phase ceiling600s; one local GPU,
CPU threads2. No full training, remote submission, successor or promotion release.
Scratch checkpoint is removed after byte verification; retain result metadata,
frozen source/config/prospective identity and measured worker/process/GPU costs.
GPU allocation wall includes CPU cases because RNG capture can initialize CUDA;
it is not exclusive kernel time. CPU cases measure CPU tensor checks/IO only.

## Adoption Rule

Require CPU wiring/continuation tests and byte-identical native checkpoint
serialization. Adopt only the shared checker, without a runtime slow/fast switch.
Do not adopt model algebra reorderings, sampler changes, fused optimizers,
checkpoint-frequency changes or asynchronous IO under this diagnostic.
Complete epochs/MAE/native platform qualification remain separate contracts.
