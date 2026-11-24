# SCNet BW-1 / DTK PyG Reference

## Known-good 2D GPS environment

The verified SCNet combination was:

```bash
module purge
module load compiler/dtk/25.04
module load sghpcdas/25.6
source /public/software/sghpc_sdk/Linux_x86_64/25.6/das/conda/etc/profile.d/conda.sh
conda activate pytorch-python3.10
```

Observed versions:

```text
PyTorch: 2.4.1+das.opt2.dtk2504
HIP:     6.1.25065
Python:  3.10.16
PyG:     2.7.0
```

Do not use a generic CUDA wheel. The alternate `pytorch241-py310` environment
had a NumPy 2 ABI mismatch in this deployment.

For pure-Python PyG, preserve platform Torch and install only under the project:

```bash
mkdir -p .scnet/site
python -m pip install --target .scnet/site --no-deps torch-geometric==2.7.0
export PYTHONPATH="$PWD/src:$PWD/.scnet/site${PYTHONPATH:+:$PYTHONPATH}"
```

## One-DCU Slurm gate

`hx1hdnormal` with one `dcu:Hygon` was validated, but account CPU and memory
limits have varied. Query the active allocation before adapting a Slurm file.
The current repository has `platforms/scnet/check_dcu_environment.py` and
`platforms/scnet/smoke_phase8_gps_dcu.slurm`; verify the selected frozen
package's internal paths and resource request before submission. The
historical 2,000-graph GPS smoke established portability, not model quality
or formal throughput.

## PyG SchNet blocker

SchNet needs `torch_cluster` for `radius_graph`. In the verified DTK runtime,
the package was absent. A user-space source build encountered:

1. a stale `INCLUDEPY` path causing `Python.h` failures;
2. an old default GCC incompatible with the extension;
3. missing `gflags` and `glog` development headers.

Although a newer GCC module can coexist with DTK, do not convert this into a
long ad hoc build chain. Request SCNet support for a `torch-cluster` package
matching exactly:

```text
partition/runtime: hx1hdnormal, compiler/dtk/25.04
PyTorch:           2.4.1+das.opt2.dtk2504 (HIP 6.1.25065)
Python:            3.10.16
PyG:               2.7.0
```

Do not claim the **PyG SchNet** path portable until that package is installed
and the full SchNet forward smoke succeeds. A separate SchNetPack 2.0.4 path
has been validated without `torch-cluster`; see `schnetpack-v204.md`. The lack
of memory-efficient attention emitted by GPS is non-fatal but must be considered
in later speed measurements.

## Rejected support-linked wheel

An isolated one-DCU probe installed
`torch_cluster-1.6.0+das.opt1.dtk24043-cp310-cp310-manylinux_2_28_x86_64.whl`
into a project-local target. Import then failed before `radius_graph` with:

```text
undefined symbol: _ZN3c1017RegisterOperatorsD1Ev
```

The package metadata reported `dtk24042`, while its filename reported
`dtk24043`. Treat this as a concrete C++ ABI mismatch against the DTK 25.04,
PyTorch 2.4.1 environment. Do not retry this wheel; require an exact
`das.opt2.dtk2504` build or a centre-provided package.

## Upload and operational notes

- Use the SCNet high-speed uploader for payloads over 2 GB or more than 1,000
  files; preserve paths and avoid spaces/non-ASCII in remote runtime paths.
- Upload graph caches rather than rebuilding ETKDG conformers for a small smoke.
  For the current Phase 8 setup these are the 2D and 3D expansion500k cache
  files under `results/phase8/`.
- Never run training directly on a login node. Use `squeue`/`sacct` and the
  job-specific logs to monitor submitted work.
- Use one DCU per process unless the selected trainer and contract have a
  validated distributed path and scaling evidence.

If a job exits at zero seconds with `RaisedSignal:53` and no logs, test the
Slurm log directory and account quota before diagnosing the DCU. A writable
foreground `srun` with an unwritable log path points to storage, not device
compatibility. Remove or move only verified regenerable data, then confirm a
write probe before resubmitting.
