# Stage-1 infrastructure failure 122249373

Job `122249373` ran for 69 seconds on DCU node `e06r4n13` and then failed while
deserializing the accepted development graph shard:

```text
ModuleNotFoundError: No module named 'molgap.pcqm_wedge'
```

The uploaded minimal source tree contained the audit module but omitted
`src/molgap/pcqm_wedge.py`, whose `WedgeData` class path is recorded in the
immutable PyTorch/PyG cache pickle. The job did not load any prediction payload,
construct a model, train, or produce scientific metrics. This is an
infrastructure-only packaging failure.

The unchanged retry packages the complete tracked `src/molgap/` tree rather
than an inferred minimal subset. Its Slurm entry point first deserializes the
accepted graph shard, verifies graph fields and all payload hashes, and writes
an atomic preflight record before Stage 1 can run.
