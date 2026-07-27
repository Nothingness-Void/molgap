# MolGap Kaggle Fusion Runner

This directory is a Kaggle Kernel deployment skeleton. It intentionally does
not contain credentials or large data files.

## Intended input Dataset

Create one private Kaggle Dataset and attach it to the Kernel. Its files should
be the already-produced frozen inputs:

- `pyg_3d_graphs_etkdg_expansion_1m.pt`
- `gps_expansion_1m_embeddings.pt`
- `gps_expansion_1m_depth9_embeddings.pt`
- `schnet_expansion_1m_embeddings.pt`

Kaggle is appropriate for the late fusion head once these inputs exist. ETKDG
construction and SchNet embedding extraction remain on SCNet/Colab.

## First GPU gate

1. Copy `kernel-metadata.json.example` to `kernel-metadata.json`.
2. Set the Kaggle account slug and private input Dataset slug.
3. Historical command after archiving: `kaggle kernels push -p scripts/phase8/archive/remote/kaggle/molgap_fusion_1m`.
4. Inspect with `kaggle kernels status <account>/molgap-fusion-1m`.
5. Download with `kaggle kernels output <account>/molgap-fusion-1m -p results/phase8/remote/kaggle`.

The first submission runs only `fusion_smoke.py`. Do not submit a full fusion
job until the input Dataset mount and GPU are verified.

## Verified P100 compatibility

On 2026-07-18 Kaggle assigned a Tesla P100 (compute capability `sm_60`) to
this private kernel. Its stock `torch 2.10.0+cu128` cannot execute on P100,
because that build excludes `sm_60`. The reusable fix is
`p100_torch_probe.py`: enable internet for the bootstrap run and install
`torch==2.7.1+cu126` from the official PyTorch CUDA 12.6 wheel index before
importing torch. The preflight then verifies `sm_60` is present and performs
an actual CUDA matrix multiplication.

The private Dataset is mounted below
`/kaggle/input/datasets/<owner>/<dataset-slug>`, not directly below
`/kaggle/input`. `fusion_smoke.py` resolves this path recursively and verifies
the three embedding payloads before any long-running job.
