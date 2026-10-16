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
