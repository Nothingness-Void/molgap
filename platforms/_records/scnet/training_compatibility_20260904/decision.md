# SCNet training compatibility probe

## Result

The visible Kunshan accelerator can accept and execute real DCU jobs, but it
cannot run the current MolGap GraphState training stack unchanged. The bounded
probe reached the platform DCU PyTorch environment and stopped before model
construction because `torch_geometric` was unavailable.

The Wuzhen/A-zone login remains reachable but exposes no Slurm partition, so a
runtime probe cannot be scheduled there with the current account.

## Evidence

- Scheduler submission probe `120816243`: `COMPLETED`, exit `0:0`, three
  seconds.
- GraphState probe `120816310`: `FAILED`, exit `1:0`, three minutes and two
  seconds on `kshdtest` node `j15r3n10`.
- Environment module: `apps/PyTorch/1.10/dtk-22.04-py37`.
- Failure: `ModuleNotFoundError: No module named 'torch_geometric'`.
- No official dataset was read and no model step ran.
- Estimated accelerator use for both probes: `0.0514` card-hours.

## Disposition

Do not submit a dataset training job with the existing A-zone Slurm templates
or the Kunshan platform module alone. A Kunshan-specific isolated runtime must
first provide compatible Python, PyG, OGB, and required scatter operators, then
pass this same synthetic forward/backward/optimizer probe. The 16 GiB device
capacity is not currently implicated.

Compact fields are in `summary.json`; raw output is retained beside this file.
