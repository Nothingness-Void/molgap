# Local/global allocation version-3 failure diagnosis

Kaggle1 kernel `nothingnessvoid/molgap-pcqm-local-global-allocation-s42`,
version 3, ended with `ERROR` after 114.2798 seconds and before any candidate
GPU preflight or training epoch.

The accelerator request succeeded: `progress.json` records two `Tesla T4`
devices and the intended candidate assignment. CPU initialization preflight
also completed with matched shared parameters and finite parameter counts:

- full-GPS: 4,891,057;
- sparse-GPS-3/6/9: 3,999,409;
- no-attention GraphState: 3,665,809.

Both inherited Linux `fork` workers then failed at the first CUDA initialization
with `torch.AcceleratorError: CUDA error: initialization error`. PyTorch/CUDA
does not support initializing an accelerator in this inherited process state.
No model produced a prediction, validation metric, checkpoint or scientific
result.

The infrastructure-only repair launches two fresh Python subprocesses and sets
each process's `CUDA_VISIBLE_DEVICES` before it imports or initializes CUDA.
Each worker independently reloads the same accepted immutable cache. The model,
data roles, geometry, target, split, seed, FP32 precision, batch, optimizer,
learning rate, schedule, patience, candidate assignment and acceptance rule do
not change.

Retained evidence root:
`platforms/_records/kaggle/training/pcqm_gap100k_local_global_allocation_seed42_v3`.

- `failure.json` SHA-256
  `046dd7f30ef4fb14c4d12be413a117a24aa6094c7084b9ea3bef9337af64e57d`;
- `initialization_preflight.json` SHA-256
  `97c2de80ade12af0c1335783735e95cb2d7afc0aa96da7975e09583a6223ddcb`;
- `progress.json` SHA-256
  `7782cca434911f0c6f2137799a39dfc853b714384431271dfe547e0dffa86dc3`;
- `worker_gpu0_failure.json` SHA-256
  `8927843795df2ed01a58927ecb2921491fe73fd36b9f36a51f2c0f808bd98a82`;
- `worker_gpu1_failure.json` SHA-256
  `40117d981c8366255abc5cd8e6f856fa0a8554eee57be04593dfb022a257cd64`;
- terminal log SHA-256
  `db2c238235af4ca33224b2f40f4f8c9401607513fa230505dc1fdc0b6e26bec1`.

Every terminal record keeps `official_validation_role_read=false` and
`test_dev_role_read=false`; no molecular-research-server path was accessed.
