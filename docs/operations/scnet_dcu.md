# SCNet BW-1 DCU Portability Gate

This guide defines a bounded portability gate before a formal DCU workload.
Live jobs and model selection are intentionally delegated to
`CURRENT_STATE.md`.

## Why a Gate Is Required

The local development environment is NVIDIA CUDA PyTorch. BW-1 uses the DTK/DCU
runtime, so the local Windows virtual environment and CUDA wheels must not be
uploaded or installed on the target. PyTorch Geometric extensions are the main
compatibility risk for the GPS and SchNet graph workloads.

## 1. Activate the Verified DTK Environment

For the allocated BW-1 partition, the verified platform combination is:

```bash
module purge
module load compiler/dtk/25.04
module load sghpcdas/25.6
source /public/software/sghpc_sdk/Linux_x86_64/25.6/das/conda/etc/profile.d/conda.sh
conda activate pytorch-python3.10
```

It provides Python 3.10, NumPy 1.24.3, and DTK-adapted
`torch 2.4.1+das.opt2.dtk2504`. Do not run `pip install torch` or install a
CUDA PyTorch wheel. The alternate `pytorch241-py310` environment has a NumPy
2 ABI mismatch and is not the chosen environment.

Install PyG in the uploaded project directory, without changing platform
PyTorch or adding binary PyG extensions:

```bash
mkdir -p .scnet/site
python -m pip install --target .scnet/site --no-deps torch-geometric==2.7.0
export PYTHONPATH="$PWD/src:$PWD/.scnet/site${PYTHONPATH:+:$PYTHONPATH}"
```

This leaves the DTK runtime unmodified. The one-DCU PyG gate must still pass
before adding any compiled PyG extension. If one is needed, ask SCNet support
for a wheel compatible with the exact DTK-adapted PyTorch build above.

## 2. Upload the Minimum Smoke-Test Payload

Upload the repository source excluding `.venv/`, local caches, and unrelated
results, plus these existing graph caches:

```text
results/phase8/pyg_2d_graphs_bond_expansion_500k.pt
results/phase8/pyg_3d_graphs_etkdg_expansion_500k.pt
```

The two caches contain ETKDG graphs and labels, so the 2,000-molecule smoke
test does not need to rebuild conformers or upload the raw 500K CSV. Keep file
and directory names ASCII with no spaces.

### Storage placement

The user home at `/public/home/$USER` has a separate 50 GB quota. The team
allocation provides 400 GB at `/work1/share/$USER`; global `df` output does not
show the user-home quota. Keep source, lightweight models, logs, and active
manifests in the project home, but move completed multi-GB graph caches and
embeddings to `/work1` with:

```bash
bash scripts/phase8/remote/scnet/migrate_project_storage.sh \
  results/phase8/<completed-run>
```

The migration uses a persistent `.partial` destination, checksum-aware rsync,
a no-difference verification pass, and an atomic completion marker. It replaces
the original directory with a symlink only after verification, so existing
project-relative paths remain valid. Confirm compute-node access with a bounded
Slurm write probe before making `/work1` the destination on a new account.
For multi-GB formal moves, submit `migrate_phase8_storage.slurm` with a dependency
on the final reader job. Do not rely on a detached login-node process: SCNet may
reap it when the SSH session closes.

If a job exits at zero seconds with `RaisedSignal:53` and no logs, test the log
directory directly. `Could not open stdout file: No space left on device` means
the user-home quota is exhausted; it is not evidence of a DCU, QOS, or account-
mapping failure.

## 3. Allocated Slurm Interface

The account has been validated on:

```text
partition: hx1hdnormal
GRES:      dcu:Hygon:1
node:      128 CPU cores, 510000 MiB, 8 DCUs
```

The account limit is 16 CPU cores per job. The partition default is 3.8 GiB
per requested CPU, so the 1-DCU scripts request 16 CPU cores and at most 59
GiB of memory.

### Compute-Node Network Boundary

BW-1 compute nodes cannot resolve external Internet hosts. Job `693894` proved
this for the PubChemQC Hugging Face endpoint: all four array tasks initialized
their local dependency and exclusion index, then failed with DNS resolution
before any candidate row was collected. Use SCNet for already-uploaded graph
construction and training only. Internet-backed PubChemQC acquisition belongs
on Kaggle (or another Internet-enabled client), followed by an explicit upload
of the resulting CSV.

## 4. Run the One-DCU Gate

Submit the template:

```bash
sbatch scripts/phase8/remote/scnet/smoke_phase8_dcu.slurm
```

The job validates DTK PyTorch, PyG pooling, and MolGap imports, then runs two
epochs each of GPS and SchNet on the first 2,000 already-built ETKDG/2D graphs.
All outputs stay under `results/scnet/`; no formal checkpoint is overwritten.

### Current Compatibility Result

The BW-1 gate has verified the full 2D GPS forward path on a DCU with finite
output. The platform environment lacks a compatible `torch-cluster`, so the
existing PyG SchNet implementation cannot use `radius_graph`. The support
wheel and source-build paths remain rejected as recorded below.

This does not block every SchNet implementation. The original SchNetPack 2.0.4
uses its `TorchNeighborList` implementation and native PyTorch tensor
operations rather than `torch-cluster`. It passed a separate DCU
forward/backward smoke test; see "SchNetPack 2.0.4 3D Portability Gate" below.
Continue to use the GPS-only gate for PyG-only work:

```bash
sbatch scripts/phase8/remote/scnet/smoke_phase8_gps_dcu.slurm
```

The existing PyG SchNet gate remains blocked until a compatible extension is
supplied. Do not replace it with a hand-written radius-graph fallback for a
formal comparison.

The GPS-only gate completed successfully on 2026-07-17 using one BW DCU and
the existing 500K 2D graph cache. It trained the 2,000-graph, two-epoch smoke
run without device or PyG errors. The first epoch took 7.2 seconds and the
second took 1.3 seconds; these figures are portability checks, not benchmark
quality results. DTK emitted a non-fatal warning that memory-efficient
attention is unavailable, so a later speed benchmark should use the actual
formal batch size and epoch count.

### Support Request for 3D

Ask SCNet for a `torch-cluster` package compiled against all of the following:

```text
partition/runtime: hx1hdnormal, compiler/dtk/25.04
PyTorch:           2.4.1+das.opt2.dtk2504 (HIP 6.1.25065)
Python:            3.10.16
PyG:               2.7.0
```

The user-level source-build attempt reached C++ compilation but exposed a
misconfigured shared Python include path, a GCC-version mismatch, and missing
`gflags`/`glog` development headers. A centre-provided wheel or DTK PyTorch
image with `torch-cluster` is more reliable than further local workarounds.

### Rejected Wheel Probe

On 2026-07-17, the support-linked candidate
`torch_cluster-1.6.0+das.opt1.dtk24043-cp310-cp310-manylinux_2_28_x86_64.whl`
was installed only into the isolated project target
`.scnet/site-torchcluster-candidate` and checked in Slurm job `692461`. It
failed at import before any model forward:

```text
OSError: torch_cluster/_version_cuda.so: undefined symbol:
_ZN3c1017RegisterOperatorsD1Ev
```

The wheel metadata also identifies itself as `dtk24042`, despite the supplied
filename ending in `dtk24043`. This is a PyTorch C++ ABI mismatch, not a Python
path issue; do not retry that wheel or use it for formal work. The isolated
target is not on the normal `PYTHONPATH` and does not affect the verified GPS
environment.

## 1M 2D Graph Build

The uploaded `data/raw/phase8_expansion_1m.csv` was validated as exactly
1,000,000 canonical-SMILES-unique molecules, including the unchanged 500K
core. Its SHA256 is
`f5df806cb6884f9698947611cb3b5eb367e6c60c0f98b985388b0d6654b6ccc1`.

Slurm job `692610` completed on 2026-07-17 using
`scripts/phase8/archive/remote/scnet/build_phase8_expansion_1m_2d.slurm`. The
remote-only cache contained 1,000,000 valid 2D graphs, zero failures, and source
indices covering 0 through 999,999 exactly once. The cache itself is not a
tracked local asset. The machine-readable validation record is
`results/phase8/expansion_1m/graph_build_report.json`.

## SchNetPack 2.0.4 3D Portability Gate

SchNetPack 2.0.4 was installed as a separate user-level target at
`.scnet/schnetpack-v204`, leaving the platform DTK PyTorch and the normal PyG
target unchanged. The target uses the platform `torch 2.4.1` and `numpy 1.24`;
packages were installed with `--no-deps` to prevent pip from replacing either.
The required `antlr4-python3-runtime==4.9.3` was supplied as a pure-Python
wheel because the configured mirror exposed only a newer incompatible runtime.

Slurm job `692685` ran the original `TorchNeighborList`, `PairwiseDistances`,
and `SchNet` modules over 2,000 cached ETKDG molecules. It completed 63
forward/backward/AdamW batches on one BW DCU with finite losses (`4.142189` to
`0.856618`); the model loop took 9.56 seconds. Peak resident memory was about
6.5 GB. The result is stored in `results/scnet/schnetpack_v204_smoke.json`.

This is a portability result only: it does not establish a 3D MAE, a fair
comparison with the existing SchNet encoder, or formal-training throughput.
Before a 500K 3D run, train and evaluate a 30K SchNetPack model on the same
ETKDG split and three targets used by the current 3D baseline. Compare MAE,
wall time, and memory before promoting this alternate implementation.

### Deferred 3D Trainability Sequence

On 2026-07-17, the 30K SchNetPack trainability job `692741` demonstrated
finite early convergence after its 128-molecule preflight `692729`. The user
then stopped job `692741` and its dependent 500K job `692754` to prioritize
full 2D training. No 3D quality conclusion or candidate checkpoint was
promoted; the preflight artifacts remain portability evidence only.

### Historical 1M Dual-GPS Sequence (2026-07-17)

The validated 1M cache was used by two full 2D continuation jobs. GPS7 job
`692797` and GPS9 job `692798` wrote independent checkpoints, metrics, and full
aligned embeddings. Their model decision is recorded under
`results/phase8/expansion_1m/`; this operations guide does not restate it.

## Promotion Criteria

Proceed only when all conditions hold:

1. The retrieved `dcu_environment.json` shows an accelerator and finite PyG
   pooling output.
2. The intended encoder smoke job finishes without an unsupported operator,
   device, or ABI error. The PyG SchNet path still requires `torch-cluster`;
   SchNetPack 2.0.4 is an independently validated alternate 3D path.
3. The metrics JSON files exist and have finite MAE values.

If the gate fails at a PyG extension, stop after collecting the environment
report and contact SCNet support with the DTK/PyTorch versions. Do not spend the
remaining allocation on CPU fallback.

## After the Gate

The current code uses one accelerator per process and has no DDP training path.
Run the first formal task on one DCU. Do not request all eight DCUs until a
separate distributed-training implementation and scaling benchmark exist.
