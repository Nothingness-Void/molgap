# Checkpoint recovery attribution

The original tail failed while calling st_size as a method; it did not
invalidate the retained scratch comparison or serialize a failed checkpoint.
This bounded CPU-only recovery reuses the corrected atomic-save/resume helper.
Its resumed update matches uninterrupted loss, parameters, BN buffers,
optimizer state and RNG exactly on the two frozen training-only batches.
The new checkpoint preserves the failed attempt's saved bytes.

This qualifies only this short CPU continuation. Native CUDA performance,
full training replay and model accuracy remain unqualified; no new training
or reference is justified merely to complete those fields. See result.json
and acceptance.json for the observed identities, counts and measured cost.
