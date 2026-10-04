# CPU runtime qualification before frozen audit recovery

The 2026-10-04 recovery qualifies the exact GPU audit bootstrap on CPU, without
mounting graph datasets, opening models or performing inference. Kaggle's parent
Python may change; uv 0.8.22 creates an isolated managed Python 3.12 environment.
The workload retains numpy 1.26.4, torch 2.4.1+cu121, torch-geometric 2.6.1,
ogb 1.3.6 and rdkit 2025.9.5. It checks package versions, CUDA build and import
compatibility only. The coordinator keeps its system interpreter.

The private CPU kernel mounts only the frozen audit input dataset; the embedded
audit entry is SHA-verified and invoked with `--environment-only`. A 30-minute
environment deadline applies. Device hours are not applicable; observed wall,
missing CPU and queue costs remain distinct. No graph roles are applicable.

Import success is not GPU calibration, numerical reproduction or scientific
acceptance. A alone accepts the source-bound CPU output, closes its RML attempt,
and may release the separate v2 frozen GPU audit. The original reproduction
barrier and fixed 90-minute GPU cap remain mandatory. ERROR stops this release
path for diagnosis; B never submits the GPU successor itself.

Implementation references: [uv Python management](https://docs.astral.sh/uv/guides/install-python/),
[uv isolated environments](https://docs.astral.sh/uv/pip/environments/),
[PyTorch retained releases](https://pytorch.org/get-started/previous-versions/).
