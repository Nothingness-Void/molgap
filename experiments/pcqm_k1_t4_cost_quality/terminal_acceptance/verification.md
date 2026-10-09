# Acceptance checks - 2026-10-10

- Explicit Kaggle3 credential adapter queried exact kernel COMPLETE once and
  retrieved version1 outputs. Raw API entry bytes match frozen source;
  kernel137853937/currentVersion1/private/T4 and two dataset mounts agree.
  API `lastRunTime` differs from retained submission chronology; it was not
  used as an execution/allocation timestamp or to change run provenance.
- CPU-only `accept_retained.py` completed with existing family artifact,
  TargetIdentityBinding, runtime-preflight and paired-bootstrap helpers.
  A first cost-log read failed because SDK exported CP936, not UTF-8. The
  explicit retained-log decode was corrected and the full inspector reran.
  No production implementation, frozen input or remote source was modified.
- Targeted tests:402 passed/3 skipped in245.04s:
  `test_experiment_workflow.py`, `test_experiment_family_workflow.py`,
  `test_same_run_replay.py`, `test_experiment_terminal.py`.
  Synthetic tests establish interfaces, not real-run RML acceptance.
- Existing owner RML validation:106 trajectories/93 evidence/25 trace
  manifests. The two new arms remain prospective ACTIVE, with no formal
  terminal transaction executed. See [identity blockers](RML_BLOCKERS.md).
- Existing-corpus `check --frozen --portable` passed. Derived files are current
  and were not manually edited. This check does not qualify the two ACTIVE arms;
  no new terminal world was created and no terminal RML rebuild is claimed.
- Desktop control-document bounds and top-level pointer tests:2 passed.
  Raw-byte CRLF evidence was preserved; Git whitespace checking with
  `core.whitespace=cr-at-eol` passed, without normalizing frozen files.

No local training/inference, new submission, retry, GPU/DCU/IMS command,
protected-role access, policy-tolerance change or production promotion.
Whole allocation cost and strict comparison/replay remain explicitly pending.
