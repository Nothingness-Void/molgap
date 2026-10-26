# Resume admission failure — 2026-10-27

Kaggle2 kernel140916841/version2 ended ERROR during actual-checkpoint resume
preflight. Its execution report contained only the failed preflight phase; it
contained no new formal-training phase. ESA reported an incomplete resume state
because the disabled EMA field was absent. The coordinator terminated its peer.
The original checkpoints and completed version1 progress were retained.

The resume-preflight loader required the same explicit `ema=None` normalization
already used by other checkpoint loaders. This was an infrastructure compatibility
repair, not a model, data, optimizer, schedule, precision or endpoint change.
The shared-validator requirement for enabled EMA was not relaxed.

The version2 bootstrap observed1304.107362469s and its overlapping allocation
ledger observed1303.689450155s. They described one allocation and must not be
summed. The ledger retained the earlier6877.277398151s invocation. Any continuation
must debit both invocations within the original32372s aggregate ceiling, including
its600s output reserve; account quota alone cannot substitute for this accounting.

Release remained blocked pending a committed, verified recovery-source repair.
The shared recovery-overlay infrastructure was under independent, unfinished
integration in the same checkout. Those unrelated changes were preserved. No
version3, fresh baseline, scientific successor, protected-role read or final
scientific/Replay claim was released by this diagnosis.

[Compact record](summary.json) pins the exact retrieved version2 evidence.

The bounded [Luna verification](verification.json) passed7 existing checks and
metadata-only inspection of both retained checkpoints. It did not directly
execute the native resume-preflight entrypoint or qualify a GPU release.
