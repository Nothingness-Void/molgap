# P8.19 2M-2D plus 1M-3D Fusion Launch

Input datasets:

- `nothingnessvoid/1m-full`
- `nothingnessvoid/molgap-2m1m-fusion-staging-20260722`

The first two kernel versions exposed Kaggle's single-file script behavior:
version 1 could not import the companion `fusion.py`, and version 2 could not
read the companion `variant.json`. No training occurred in either version.
`package_variants.py` now injects the canonical `FusionHead` and a literal
variant into `code_file`; version 3 passed those startup points and aligned all
997,445 SchNet rows, but Kaggle assigned P100 GPUs while its default PyTorch
image contained only `sm_70+` kernels. The first compatibility retry installed
torch 2.7.1 without its cuSPARSELt runtime and failed explicitly before
training. The active retry installs the matching runtime, checks a real CUDA
matrix multiplication, and has remained running beyond all earlier failure
points.

Kernels:

- `nothingnessvoid/molgap-2m-2d-1m-3d-fusion-coverage2m`
- `nothingnessvoid/molgap-2m-2d-1m-3d-fusion-hard20k`
- `nothingnessvoid/molgap-p819-multi2d-1m3d`

The first two controls completed and their raw downloads are preserved under
their versioned `complete_v*` directories. The original multi2d slug entered a
Kaggle ghost state: it was absent from status and account listings while create
requests returned `Notebook not found`. The independent P8.19 slug above was
therefore used without overwriting either completed control. It also completed,
and all three raw downloads are retained. Scientific acceptance and the final
negative decision are recorded in
`results/phase8/multi2d_2m_1m3d_fusion/decision.md`.

The future sealed 20K is not mounted or opened by these controls.
