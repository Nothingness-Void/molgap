# Durable first-segment acceptance — reviewed2026-10-15

The two bound Kaggle3 version3 invocations were observed COMPLETE. All four
arms had stopped normally at the acknowledged pass10 boundary, not at the
frozen60pass scientific endpoint. The [compact acceptance record](segment_pass10_acceptance.json)
retained the independent results and exact artifact pointers.

## Accepted evidence

The review used version-specific metadata and hash-pinned streamed payloads;
it did not enumerate or download worker environments. Source/package, recipe,
fixed-data, initialization, arm, physical receipt and trajectory identities
were checked against the original release. Both assigned arms in each slot
passed `accept_campaign_segment`.

The native repeat and serialized-resume proof tensors were independently
rechecked, including model, optimizer, scheduler, EMA and isolated CUDA RNG.
Zero-start parent agreement, mechanism gradients and resource headroom were
checked from the retained native certificates. Prefix checks verified complete
live-train/live-dev/EMA-dev observations, exposure and paused cursor, finite
FP32 state, row/target hashes, best-EMA selection and saved-prediction MAE.
Each arm's portable export also passed the original family resume validator.
No actual retained-state GPU continuation was executed by this review.

The local working tree contained independent scientific-module edits. Those
edits were preserved. Acceptance loaded scientific dependencies from the
original verified source archive, with the reviewed read-only segment consumer;
it did not disable source checks or substitute edited model code. Consumer
and source provenance were retained in the per-slot reports.

## Cost and interpretation limits

The compact record counted each Python bootstrap allocation envelope once,
including idle allocated devices and its short finalization tail beyond the
runtime ledger, plus the separately retained version1/2 failed-attempt costs.
Overlapping bootstrap and runtime windows were not added together. Provisioning
before Python and queue intervals remained unmeasured, not zero. Remaining
campaign ceiling was not represented as available account quota.

The prediction metrics described only the first ten passes. No candidate was
declared a winner or loser, no immutable final reference was accepted, and no
training Replay qualification or production nomination was claimed. The
remaining50passes, actual native continuation, final endpoint comparisons and
terminal V5/RML closure had not been performed. Four prospective chains remained
incomplete; they were not finalized as completed scientific experiments.

## Verification and scope

The focused segment/transport regression passed49tests; repository layout
coverage passed26tests. RML validation and frozen-derived checks passed.
This review performed no local model forward,
training, graph-cache loading, protected-role evaluation, new remote submission
or private checkpoint publication. The user's `tests/test_research_memory.py`
bytes were preserved.
