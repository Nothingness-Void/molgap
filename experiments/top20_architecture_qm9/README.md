# QM9 Architecture Evidence

This directory keeps the accepted pure-2D R3 validation evidence and the
literature audit used to choose the PCQM architecture questions. The original
TGT transfer screen, PairGPS R2, the untriggered R4 fallback, and R5--R10 are
closed or inactive; their complete reproducible assets are preserved on the
`archive` branch and indexed by
`experiments/_closed/qm9_top20_archive_index.md`.

## Retained comparator

The validation-only R3 tournament selected persistent real-bond EdgeState
Structural GPS as its pure-2D comparator. Its dated decision is
`pair_gps_2d_r3_decision.md`, and its inference-free acceptance is
`results/pure2d_r3_validation_acceptance.json`. The QM9 test role remains
sealed for this server-side line; the result does not by itself authorize a
PCQM full-data run.

## Literature evidence

`top20_audit.md` records the transferable operations from the official
top-20 snapshot. It is retained as design evidence only. New architecture
questions belong under `experiments/pcqm_gap_architecture/` and must follow
the active contract in `CURRENT_STATE.md` and `ROADMAP.md`.
