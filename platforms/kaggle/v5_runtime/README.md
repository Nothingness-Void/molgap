# Kaggle V5 Desktop Runtime

This adapter creates a deterministic private source snapshot for Kaggle desktop
work. The runtime contains the tracked `src/molgap` Python package, dependency
metadata, and the V5 operating/V4 comparison constraints. It deliberately does
not include a scientific run spec, protected data, credentials, or training
authorization.

Future kernels mount this runtime plus one accepted fixed-data role and carry
only a small experiment-specific entrypoint and immutable run spec. Each run
must still perform its own hardware/runtime preflight and bind the resulting
certificate before training.

After publication, download the complete expanded dataset and run
`accept_runtime.py`. Kaggle expands the deterministic archive into
`molgap_runtime/`; acceptance therefore verifies every expanded file against
`SOURCE_FILES.json` instead of trusting dataset readiness alone.
