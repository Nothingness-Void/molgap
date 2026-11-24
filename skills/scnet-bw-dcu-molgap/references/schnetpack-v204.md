# SchNetPack 2.0.4 on BW-1 DTK

Use this alternate 3D path only with cached ETKDG coordinates. It uses
SchNetPack's `TorchNeighborList`, native PyTorch operations, and no
`torch_cluster`; it is not checkpoint-compatible with MolGap's PyG
`SchNetWrapper`.

## Verified base environment

```bash
module purge
module load compiler/dtk/25.04
module load sghpcdas/25.6
source /public/software/sghpc_sdk/Linux_x86_64/25.6/das/conda/etc/profile.d/conda.sh
conda activate pytorch-python3.10
```

Keep platform `torch 2.4.1` and `numpy 1.24` unchanged. Never run a normal
dependency-resolving `pip install schnetpack`, because it can replace them.

## Project-local bootstrap

Use variables rather than account-specific paths:

```bash
PROJECT_DIR=/path/to/molgap
SPK_TARGET="$PROJECT_DIR/.scnet/schnetpack-v204"
PYG_TARGET="$PROJECT_DIR/.scnet/site"
mkdir -p "$SPK_TARGET" "$PYG_TARGET"
python -m pip install --target "$PYG_TARGET" --no-deps torch-geometric==2.7.0
python -m pip install --target "$PYG_TARGET" --no-deps rdkit==2026.03.2
python -m pip install --target "$SPK_TARGET" --no-deps schnetpack==2.0.4 ase==3.22.1 matscipy==1.1.0 h5py==3.11.0 hydra-core==1.3.2 omegaconf==2.3.0 pytorch-lightning==2.0.9.post0 torchmetrics==1.0.1
export PYTHONPATH="$PROJECT_DIR/src:$SPK_TARGET:$PYG_TARGET${PYTHONPATH:+:$PYTHONPATH}"
export CRYPTOGRAPHY_OPENSSL_NO_LEGACY=1
```

Install only the missing pure-Python transitive packages with `--target` and
`--no-deps`; inspect each wheel before adding it. In the validated deployment,
OmegaConf required `antlr4-python3-runtime==4.9.3`, while the configured mirror
offered only 4.13.2. Transfer or build the upstream 4.9.3 pure-Python wheel on
another machine, then install it with `--no-deps --upgrade --force-reinstall`.

Validate imports without reading `schnetpack.__version__` (the package does not
expose it):

```bash
python -c "import schnetpack; from schnetpack.representation import SchNet; from schnetpack.transform import TorchNeighborList; print('ok')"
```

## Historical qualification and new submissions

The earlier SchNetPack route used a 2,000-graph backward smoke, a 128-graph
one-epoch preflight, a 30K trainability run, then a possible 500K comparison.
Those Slurm files now live under `experiments/_closed/remote_scripts/scnet/`;
they are historical evidence, not a current submission sequence. For a new
run, follow its owning experiment contract and choose the smallest gate that
answers its question. A portability smoke alone does not authorize a 500K
quality claim or model promotion.

On a new account, confirm the compute-node runtime, a real SchNetPack
forward/backward operation, finite outputs, checkpoint write, and split/input
identity before formal training. Set `PYTHONPATH` and
`CRYPTOGRAPHY_OPENSSL_NO_LEGACY` in the selected Slurm adapter, request only
validated resources, and keep output/error logs distinct. Do not run training
on the login node.

## Known outcomes and stopping rules

- PyG `torch_cluster-1.6.0+das.opt1.dtk24043` failed with an undefined
  `RegisterOperators` C++ symbol; it is an ABI mismatch and must not be retried.
- SchNetPack 2.0.4 completed a BW DCU 2,000-graph forward/backward smoke with
  finite loss. This validates portability only, not accuracy.
- If the owning contract uses a 30K trainability gate, a numerical failure or
  missing metrics/checkpoint stops that route before a larger allocation.
- A completed 500K SchNetPack checkpoint remains an alternate encoder. Extract
  its aligned embeddings and train a new fusion head before comparing it with
  the production routed-v4 hybrid; never substitute it into existing PyG
  fusion checkpoints.
