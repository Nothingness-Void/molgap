# Lossless proof-reader repair verification — 2026-10-25

The coordinator changed `modern2d_execution_proof.verify_repair_proof` to reuse
`native_proof_storage.load_proof`. Exact allowed state names and report SHA,
state SHA, resume completeness and numerical comparison remain enforced.
Neither model mathematics nor a scientific recipe changed.

An independent configured Luna worker ran these existing workflow checks:

- `test_modern2d_formal_training.py::test_formal_actual_session_resume_full_state_and_readonly_dev`
- `test_family_training_runner.py::test_native_proof_storage_round_trip_and_interrupted_retirement`
- `test_modern2d_research_workflow.py::test_prepared_preflight_requires_fresh_accepted_native_reference`

Result:7 passed,0 failed,0 skipped in19.40s. Generated CPU PPGT/3 and ESA/4
Sessions retained4 updates/512 presentations; disk resume from2 updates/256
matched continuous state. The real execution-proof reader accepted loose then
packed/retired bytes with zero model/optimizer difference and rejected a wrong
original SHA. Storage interrupted-retirement and missing/wrong accepted-native
reference guards remained effective. Two CPU pin-memory warnings were expected.

Report: `E:/MolGap/scratch/esa-proof-repair-20261025/focused-47d7fbf51636458793de3958d9c992f0/junit.xml`.
These are checkout CPU fixtures, not a new native certificate or completed
training. A retry must repeat native admission under its exact packaged source.
