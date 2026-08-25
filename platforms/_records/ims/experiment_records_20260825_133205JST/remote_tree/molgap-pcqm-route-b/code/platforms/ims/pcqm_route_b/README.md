# PCQM Route B on IMS

This adapter runs the Track B PCQM precision encoders on the IMS/RCCS A100
queue. Model and checkpoint behavior remains owned by
`src/molgap/pcqm_route_b_training.py`.

All remote files must stay below:

`/lustre/home/users/sm2/chou/molgap-pcqm-route-b`

## Immutable inputs

- `inputs/pcqm_route_b_1m/`: accepted aligned GPS/primary/secondary graph views.
- `warmstarts/`: the four frozen warm-start checkpoints.
- `wheels/`: the offline Python 3.10 CUDA 12.6/PyG wheel set.
- `code/`: a repository snapshot containing `src/molgap`.

The IMS operating-system ABI cannot load the published `torch_cluster` wheel.
This adapter therefore places the versioned
`molgap_portable_torch_radius_v1` shim first on `PYTHONPATH`. It implements the
same cutoff, flow, self-loop, and 32-neighbor contract with vectorized Torch
operations. Every SchNet checkpoint records this radius backend.

## Execution order

1. Submit `setup_env.pbs` on a CPU node.
2. Submit `accept_inputs.pbs` on a CPU node and require every transferred graph
   shard to match the accepted local manifest.
3. Submit `preflight.pbs` for one real forward/backward/checkpoint step through
   all four encoders on one A100.
4. Accept all four preflight JSON and checkpoint files.
5. Submit `train_encoder.pbs` once per encoder with
   `jsub -q H -v ENCODER_NAME=...`.
6. Accept each atomic completion manifest and embedding manifest before fusion.

Training outputs are written directly to Lustre under `outputs/<encoder>/`.
`last.pt`, `progress.json`, and embedding part manifests make interrupted jobs
resumable without transient node storage.

Supported encoder names:

- `gps9`
- `gps11_160`
- `primary_schnet`
- `augmented_schnet`
