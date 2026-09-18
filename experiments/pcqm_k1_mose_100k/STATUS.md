# Status

Implementation and V5 protocol are committed and pushed on `molgap-server`.
Private Kaggle1 source dataset
`nothingnessvoid/molgap-pcqm-k1-mose-source` is `ready` at source commit
`12c62e476e174e9f204977e218fbf85334958045`.

Kaggle1 CPU kernel `nothingnessvoid/molgap-pcqm-k1-mose-cache-prep` version 1
failed before graph loading because the CPU image did not provide
`torch_geometric`. This was an infrastructure failure, not a scientific
result. No model was trained and no sealed role was read.

The only covered retry pinned the pure-Python `torch-geometric==2.6.1` wheel by
SHA-256 and completed as version 2. Local no-model acceptance passed all
identity, row-count, source-range, shard-hash, aggregate-hash and sealed-role
checks. The accepted cache has 150,000 rows, 2,090,775 nodes and 31 channels;
its aggregate SHA-256 is
`5d949f90a35aea5001d2f6438c916f92cab45d6c5860ba6ccf5777aac1c897e3`.
The immutable private dataset is
`nothingnessvoid/molgap-pcqm-k1-mose-cache-v1`. Acceptance evidence is
`results/cache_acceptance_v2.json`.

After confirming that no other recent Kaggle1 GPU training remained active,
the single authorized seed-42 screen
`nothingnessvoid/molgap-pcqm-k1-mose-s42` version 1 was submitted. It stopped
at accelerator preflight before model training because the entry point used a
combined strict CUDA guard without the established P100-compatible PyTorch
repair. This is an infrastructure failure, not a scientific result; compact
evidence is `results/gpu_v1_infrastructure_failure.json`.

The covered repair pins one assigned device before importing PyTorch, probes
whether the bundled wheel supports that accelerator, and installs the pinned
PyTorch 2.4.1 cu121 wheel only when required. It changes no data, model,
initialization, seed, precision, batch, optimizer, schedule, exposure or gate.
Kernel version 2 was submitted but Kaggle allocated a Tesla T4 despite the P100
request. The repaired compatibility probe passed; the remaining P100-name
guard then stopped the process before model training. This is a second
infrastructure-only outcome, recorded in
`results/gpu_v2_allocation_failure.json`.

The screening policy treats platform and accelerator as provenance when the
scientific contract matches and the runtime issues an accepted deterministic
calibration certificate. The next covered repair therefore requests T4,
isolates one visible device and removes only the accelerator-name guard. It
does not change data, model, initialization, seed, precision, batch, optimizer,
schedule, exposure or gate. Kernel version 3 reached the assigned T4 but then
stopped before training because the entry point asserted the no-TF32 contract
without first setting both PyTorch backend flags to false. Evidence is
`results/gpu_v3_tf32_preflight_failure.json`.

The final covered infrastructure repair explicitly disables CUDA matmul and
cuDNN TF32 before retaining the same assertion. No scientific-contract field
changes. Kernel version 4 passed all three earlier infrastructure preflights,
then exposed an implementation error while constructing the candidate for the
runtime certificate: OGB `AtomEncoder` has no `out_features` attribute. Model
training still did not start. Evidence is
`results/gpu_v4_encoder_interface_failure.json`.

The covered code repair uses the already frozen `HIDDEN_CHANNELS=192`
architecture constant rather than introspecting `AtomEncoder`. It preserves
the intended tensor shapes and parameter identity. The private source dataset
must be repackaged from the repaired commit before any retry. No additional
seed, protected role, 500K job or successor is authorized.
