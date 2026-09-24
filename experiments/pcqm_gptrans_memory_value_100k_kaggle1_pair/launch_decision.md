# Persistent-pair readback 100K release decision

Decision date: 2026-09-24 (Asia/Tokyo).

The GPTrans propagation-flow ablation retained both pair-to-node readback and
cross-layer pair recurrence as useful paths, but did not test whether readback
should consume the accumulated pair memory. `memory_value` changes precisely
that path and leaves node attention, relation update and training recipe fixed.
The Kaggle1 control is scientifically required for a same-allocation paired
comparison and for the new prospective same-run replay binding. Existing
historical GPTrans evidence remains contextual.

This releases one Kaggle1 T4x2 100K attempt after every gate in `protocol.md`
passes. It does not release a retry on stale status, another seed, a scale
step or any protected role. Remote completion will require independent
per-arm acceptance and a dated scientific decision.
