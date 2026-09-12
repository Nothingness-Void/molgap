# Full-train K1 and GPTrans-T comparison

This experiment trains the frozen Neural-Atom K1 and GPTrans-T candidates on
the same accepted 3,378,606-row official training role. Each receives exactly
20,000,000 sample presentations, FP32, seed 42, batch 128, and no partial
batches. Separate outputs are retained; no validation/test role is read.

- K1 contract: `../pcqm_k1_full/training_contract.json`
- GPTrans-T contract: `training_contract.json`
- Execution rationale and fusion gate: `protocol.md`
- CLI entrypoints: `../pcqm_k1_full/run.py` and `run_gptrans_full.py`
- PBS templates: `jobs/`

Full training starts only after each model's A100 preflight passes. Every
checkpoint is atomic and contains the exact resumable cursor and RNG state.
Fusion weights cannot be fit on in-sample predictions. A learned blender needs
an independent calibration role and an untouched evaluation partition. If no
such calibration role is authorized, evaluate only the fixed 50:50 average.
