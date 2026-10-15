# IMS Adapter

This directory contains scheduler and environment adapters for workloads on the
IMS molecular research server, including PCQM Route B and secondary-conformer
construction.

Project writes must remain under the approved user project root. Reusable model
logic stays in `src/molgap/`, live status stays in `CURRENT_STATE.md`, and
retrieved outputs belong in `platforms/_records/ims/`.

The fixed official PCQM4Mv2 data views used across future scale experiments are
owned by `pcqm_fixed_datasets/`. They reuse accepted graph payloads and freeze
source indices, roles, feature schema, and hashes without rebuilding graphs.

## Shared family worker

The selectively imported [V4 submit adapter](v4_submit/README.md) owns staging
and PBS submission. [run_family_training.py](run_family_training.py) only isolates
a scheduler-assigned GPU and invokes the existing shared worker. It contains no
training loop. See the [family examples](../../docs/operations/FAMILY_TRAINING_EXAMPLES.md)
for explicit runtime and prepared-input boundaries; local fixtures do not qualify
an IMS deployment or job.
