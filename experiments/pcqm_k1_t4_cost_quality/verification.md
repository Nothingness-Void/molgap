# Parent verification - 2026-10-09

The final targeted suite returned210 passed: shared experiment/family workflows,
K1 training, parent release, native T4 acceptance and same-run replay.
`check --frozen --portable` passed after the prospective pair was committed.
Tests used the project virtual environment and this owner's `src` path.
One broader test invocation initially named a nonexistent test file; it executed
no tests and was replaced by the correctly named suite. No training was run locally.

The profile's real metadata acceptance/finalization is separate from these tests;
see [its verification](profile/verification.md). The parent release pins the
canonical finalized clean-fit terminal input, not a differently serialized
equivalent JSON export. Raw API source bytes were verified independently of the
SDK's Windows newline conversion.

After these independent checks, one bounded scheduler/output reconciliation
still observed RUNNING with no currently retrievable output files. See
[scheduler](scheduler_post_checks.json), [retrieval](output_post_checks.json).
This is not an epoch count, runtime qualification, zero-progress diagnosis or
training endpoint. No further polling/heartbeat/retry was created.

The [STATUS](STATUS.md) and frozen protocol govern the next reconciliation.
Actual terminal quality, cost and strict replay qualification remain pending.
Desktop's unrelated historical layout debt is recorded separately, not hidden by
changing frozen evidence or weakening these checks.
