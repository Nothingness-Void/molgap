# Configured screen CPU release, 2026-10-19

Main reviewed the user-supplied configured infrastructure, authored the human
configuration and manifest-provenance fix, and prepared the actual frozen run.
Luna alone executed tests on synthetic/fake-indexed fixtures, one CPU thread,
CUDA hidden, without recursive delegation or real data/model inference.

- Original targeted run:33 passed,4 expected CPU warnings,312.18 seconds, exit0.
- Post-fix fresh run:25 passed,270.07 seconds, exit0. Both PPGT/3 and ESA/4
  retained the noncanonical accepted binding and passed offline preparation,
  source-bound native context and cross-HEAD continuation checks. Historical
  formal-release rejection and all-arm preflight barrier regressions passed.
- These overlapping batches are separate, not58 distinct tests.

Reports remain at E:/MolGap/artifacts/modern2d-new-screen-test-20261019/FINAL_REPORT.md
and E:/MolGap/artifacts/modern2d-new-screen-test-20261019-post-fix/REPORT.md,
with original logs and JUnit XML beside them. Luna changed no repository files.

Actual preparation of repaired local run modern2d-configured-100k-s42-v2/v2
completed in41.3593 seconds, with37.0110 seconds inside the workflow owner.
Both committed recipe/input/role/target and frozen initialization pins passed;
release status was LOCAL_RELEASE_INPUTS_VERIFIED with no errors. The historical
local-only v1 refusal remains retained rather than overwritten. Generated
compiler records are not hand-edited or committed; source/package hashes and
each arm's prospective RML records bind the release.

CPU regression and local preparation do not establish actual GPU identity,
numerical equivalence/resume on T4, throughput, model quality, billing, strict
causality or Replay admission. Native all-arm qualification is still required
before either trainer starts. Unrelated user edits, global derived RML and the
dirty user test_research_memory.py were preserved.
