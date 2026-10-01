# Retained K1 reference binding

This directory binds the reference arm of authenticated COMPLETE Kaggle3
`nvoid912/molgap-k1-v4-ssma-accuracy-100k-s42-v1`, version 1, to the new
width256 comparison. It does not close the original SSMA research question.
Original source, recipe and output hashes remain authoritative.

`build_binding.py` delegates mechanical inspection, runtime qualification and
comparison validation to shared owners. It reads retained artifacts on CPU;
it does not train or perform checkpoint inference. Run with the selected
checkout's `PYTHONPATH` and project Python. `--candidate-source` binds the final
frozen candidate source and writes its prelaunch comparison.

The first inspection receipt is preserved as
`mechanical_acceptance_initial_encoding_blocked.json`. The original K1 contract
uses raw little-endian float32 development target bytes (`a1c0b711...`); the
generic inspector originally converted them to float64 (`7b527c57...`) and
therefore rejected the identical retained targets. The shared inspector repair
uses adapter-aware hashing for `k1-screen-v1`, preserving the frozen contract.
This repair changes acceptance encoding, not reference data or training.

`target_transform.json` uses the original immutable source's accepted fp32
training transform and train-target checksum. No development statistics are
used. `cost_records.json` preserves measured T4 process/device seconds and
unknown CPU/queue cost. Training stochasticity remains unavailable for one seed.
