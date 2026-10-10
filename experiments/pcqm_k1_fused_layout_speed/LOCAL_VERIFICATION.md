# Local verification scope

Project Python: D:/文档/molgap/.venv/Scripts/python.exe, selected source via
PYTHONPATH. CPU checks use CUDA_VISIBLE_DEVICES=-1.

Before GPU: test_k1_local_speed.py, test_k1_execution_layout.py and
test_k1_screen_training.py:43 passed,1 warning. A later CPU-only acceptance
test brought the same set to44 passed. Source guard hardening plus the
v4_runtime suite:55 passed,1 warning. These synthetic/library tests do not
certify GPU next-step resume, quality, native T4/A100 or registered release.

Real local run: immutable113a20ce source2 bundle, RTX5060, both arms10ep,
15620 total optimizer steps. Worker exited0. Original TRAIN100K cache and full
initial state verified on CPU before CUDA. No development/official inputs.
Terminal checkpoint CPU inspection verifies finite model tensors, scheduler10
and optimizer7810 for initialized states. Raw timing sequence and original
cosine40 LR prefix reproduce the summary. No model inference is part of closure.

Existing finalizer published both terminal RML records, role/cost events and
canonical traces. Both traces are explicitly ineligible for scientific replay.
New worktrees did not inherit two ignored historical best_model.pt files.
Copied their immutable weight bytes from retained desktop custody; the existing
RML verifier checked their bound hashes. No historical model execution, label
reading, evidence rewriting or protected-role use occurred.

See [closure receipts](closure_receipts.json) for finalizer identities and
[result](terminal_decision.md) for measured windows and exclusions.
