# Shared trace publication correction

Dated2026-10-21 JST; prepared on the K1 extended-exposure owner. The previous
structural-bias/norm acceptance retained null LR/checkpoint histories. This
correction changes publication for new frozen payloads only.

- Existing FamilyTrainingSession observes the actual optimizer LR before each
  update and forwards the last observed rate with completed-pass metrics.
- Existing SessionOutputs defers every pass row to its existing checkpoint-last
  transaction, binding actual checkpoint bytes before atomic publication.
- Process wall is measured for every mode. Device timing still requires its
  explicit policy; a process timer is not silently labeled allocated-device cost.
- Model updates, optimizer/scheduler/RNG/EMA and evaluation counts are unchanged.
  Historical payloads and traces are not edited. This is not terminal acceptance
  or a promise that a future run will qualify for strict replay.

Focused synthetic coverage:79 distinct cases passed after correcting four new
test assertions to preserve permitted sparse non-development partial-terminal
rows. The initial batch had75 passes/four assertion failures; only those four
were rerun and passed. Existing checkpoint recovery, exact resumed state/RNG,
logging parity and archived Post-LN surgery tests are included. No real graph
training or protected-role access was used for these tests. Logs/JUnit remain
under D:/molgap-artifacts/tests/k1-norm-exposure-20261021/.

The K1 addon is reused from d6058bc57fc92b6bcb336ddac13fca6846ef4d95.
Its reviewed single policy-backed slot binding is selectively restored while
retaining the current shared-family projection; unrelated rejected modules are
not imported. Deployment remains the owning experiment's platform release.
