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
