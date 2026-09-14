# K1 full pipeline hardening decision

On 2026-09-12, the pre-execution audit found that directly scaling the earlier
500K loop would overtrain by 6.76x, use partial tail batches, lack exact resume
state, omit target/config/runtime identity from the saved model, leave A100 TF32
implicit, and measure memory without an optimizer step. The old 500K result was
not changed; it remains valid under its original paired contract.

The replacement full-role contract in `training_contract.json` fixed those
execution defects while preserving the K1 architecture. It uses exact
sample-exposure steps, full 128-graph batches, strict FP32 determinism, atomic
step-level resume, optimizer-inclusive resource gates, complete runtime/source/
data provenance, a self-contained model bundle, and no-inference acceptance.

The separate comparison-policy defect was also closed: future candidates use
one immutable benchmark reference and reusable per-runtime calibration rather
than retraining a baseline in every task. Cross-platform comparison is allowed
only when the complete scientific contract matches.

This was an infrastructure decision. It produced no model metric and granted no
full-run, official-validation, test-dev, or submission authority.

The complete issue-to-fix mapping is machine-readable in
`audit_checklist.json`; the frozen architecture is additionally locked by its
seed-42 initial-state hash.

On 2026-09-13, a pre-launch runtime audit found two stale source-file hashes in
the machine contract. The values were corrected to the canonical LF hashes of
the files at the recorded architecture commit; source hashing now normalizes
Windows CRLF checkout endings before comparison. No model, optimizer, data,
sample-exposure, or acceptance rule changed. The exact IMS runtime reproduced
the frozen 3,658,817 parameter count and seed-42 initialization hash.
