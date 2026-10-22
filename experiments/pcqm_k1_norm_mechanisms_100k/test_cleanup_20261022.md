# Applied recovery test audit — 2026-10-22

Implements the [124-case audit](test_audit_20261022.md). That dated audit remains
an assessment of the original suite, not a count of the edited suite.

- Reduce the budget Cartesian product from 48 to 11 explicit cases: retain
  every limit, both reserve branches, and installation/runtime failure paths.
- Delete the function-object-identity assertion; it established no loader
  behavior.
- Corrupt the last repair manifest member and check all original files after
  rejection. A validate-and-write-one-at-a-time implementation can no longer
  pass the original/replacement corruption cases.
- Consolidate the two canonical-record edits into one parameterized test and
  state its actual proof: changed frozen bytes are rejected. This does not
  independently prove the ACTIVE-state predicate.
- Name local release and historical-work assertions according to their scope;
  fake workers do not prove native training or POST prevention.
- Add one regression through the actual resume-probe wrapper with populated
  CPU AdamW state. It demonstrates the step-tensor alias and requires both
  scratch restores to start at the retained step without changing disk bytes.

The continuation repair deep-copies retained state for each scratch probe.
The existing repair attestation permits only that exact additional AST change
and the existing source verifier repair. Model and formal training semantics
remain pinned by the original package.

For this change, select only budget (11), overlay rejection (4), sidecar race
(2), changed checkpoint (1), probe isolation (1), and canonical-byte rejection
(2): 21 cases. The remaining scoped integration tests should run when their
owners change, not on every unrelated submission. Real Kaggle admission and
formal cursor advancement are required separately.

Executed once: **21 passed in 55.63 seconds**, exit 0. Collection only confirmed
87 cases across the original four files plus the new one-case regression
(124 - 37 - 1 + 1); it took 4.20 seconds. No broad suite was rerun. Logs are
retained under `D:/molgap-scratch/resume-probe-focused-final-cf6329fd62d140d58da2aa03bbcee3fc/`.
These results establish the listed local behaviors, not native T4 admission.
