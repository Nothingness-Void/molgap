# K1 edge-conditioned slot on the fixed PCQM-100K screen

This experiment asks one isolated information-flow question: can persistent
real-bond state improve K1's atom-to-slot selection without adding a slot,
global layer, geometry input, or target-space correction?

- `protocol.md` — frozen scientific contract and gate.
- `training_contract.json` — machine-readable v4 contract.
- `STATUS.md` — operational state.
- `accept.py` — saved-artifact-only acceptance.
- `package_source.py` — deterministic private Kaggle source package.
- `p100_candidate/` — one seed-42 Kaggle candidate launcher.

The candidate is compared directly with the immutable K1-v4 reference under
the same fixed PCQM-100K data identity. Large outputs remain under
`platforms/_records/kaggle/training/` and are not committed.

Local work is limited to static checks and no model inference.
