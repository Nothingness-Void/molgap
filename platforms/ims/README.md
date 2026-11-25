# IMS Adapter

This directory contains scheduler and environment adapters for NVIDIA A100 and
CPU workloads on the IMS molecular research server, including PCQM Route B and
secondary-conformer construction. `A100` is the canonical accelerator label in
plans and status records; exact strings such as `NVIDIA A100-SXM4-80GB` remain
only in runtime certificates and measured historical evidence.

Project writes must remain under the approved user project root. Reusable model
logic stays in `src/molgap/`, live status stays in `CURRENT_STATE.md`, and
retrieved outputs belong in `platforms/_records/ims/`.

The fixed official PCQM4Mv2 data views used across future scale experiments are
owned by `pcqm_fixed_datasets/`. They reuse accepted graph payloads and freeze
source indices, roles, feature schema, and hashes without rebuilding graphs.

Committed V4 model variants use `v4_submit/` for immutable source packaging,
fixed-dataset lookup, fail-closed staging, preflight dependencies, and atomic
submission records. Scientific contracts and checkpoint semantics remain with
the owning experiment.

Shared K1/GPTrans uses [the same family worker](run_family_training.py) with an
explicit index inside scheduler-assigned `CUDA_VISIBLE_DEVICES`. This is a thin
entrypoint, not a submitter. See the [prepared bundle boundary](../../docs/operations/FAMILY_TRAINING_EXAMPLES.md#sandbox-to-package-to-platform):
neutral sidecar transport, native preflight and actual PBS execution require
their owning IMS release; the Git-archive uploader does not stage arbitrary bundles.

Standalone models with a frozen recipe skip that package route and use `family_recipe_run.py`
(see the [V4 resume notes](v4_submit/README.md#standalone-family-models-and-resume)).

## Live Queue & Run Dashboard

Read-only IMS queue, point-limit, and `chou` V4 training monitor (strictly under `/lustre/home/users/sm2/chou`):

- CLI table: `.venv\Scripts\python.exe -S platforms\ims\check_queue.py` (or `--watch 60`)
- Agent JSON: `.venv\Scripts\python.exe -S platforms\ims\check_queue.py --json`
- Local web dashboard (`127.0.0.1` only): double-click `platforms\ims\open_ims_dashboard.bat` or run `.venv\Scripts\python.exe -S platforms\ims\check_queue.py --web --open`

Search by job ID or name, then narrow the list by status. The snapshot view
searches job IDs. The header shows the age of the displayed data and can copy
a summary of queue demand and remaining group points.

Always-on snapshot deployment, access controls and rollback: [deployment guide](dashboard/README.md).
