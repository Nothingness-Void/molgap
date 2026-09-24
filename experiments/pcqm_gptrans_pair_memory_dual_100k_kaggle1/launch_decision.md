# Pair-memory dual-candidate 100K release decision

Decision date: 2026-09-24 (Asia/Tokyo).

The V5 GPTrans-T 100K reference is already accepted, so no new baseline run is
authorized. The two new mechanisms are closely related readback variants with
the same frozen dataset, initialization and training recipe. `memory_value`
serves as the same-run relative reference for `memory_message`; both are new
experiments with independent RML replay outcomes. Their results can change
whether pair memory is taken to a separately gated scale experiment.

Release one Kaggle1 T4x2 attempt only after all checks in `protocol.md` pass.
This decision does not authorize a retry on stale status, another seed, 500K,
full-scale training, official evaluation or protected-role use.
