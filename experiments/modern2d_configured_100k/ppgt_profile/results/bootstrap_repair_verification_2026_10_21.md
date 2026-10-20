# Bootstrap repair verification — 2026-10-21

Verdict: **CPU_REGRESSION_ACCEPTED**, not native profiling acceptance.

One configured Luna leaf executed the three metadata suites directly in one
pytest invocation using the project Python3.11.15. The JUnit record contained
35 tests,35 passed,0 failures/errors/skips,4.536 seconds, process exit0:

- `tests/test_modern2d_bootstrap_budget.py`:6.
- `tests/test_modern2d_formal_release.py`:21.
- `tests/test_kaggle_bootstrap_server.py`:8.

The repaired bootstrap selected the pinned family-shared environment for one
PPGT arm, one ESA arm, a mixed pair or a same-family pair. It read the hashed
authority and allocation budget before runtime preparation; the one-arm
profile retained1800 seconds rather than16200. Existing role/recipe gates,
fresh-output protection and root error/accounting behavior were retained.
The stale formal-dispatch fixture was updated to supply a real hash-pinned
authority, without weakening production validation.

The invocation used `--noconftest -p no:cacheprovider -p ppgt_test_temp_acl`.
The external test-only plugin allowed inherited directory ACLs for mkdir0700
inside one unique temporary root, avoiding the Windows restricted-token temp
setup failure. It did not alter assertions, application code or global ACLs.
This does not establish success without that host-only workaround.

The raw log/JUnit and workaround observation were retained at
`E:/MolGap/scratch/ppgt-bootstrap-fix-20261020/accepted-20261021/`.
Log SHA256:`4b775b43b8a715e9f390f90c8cfc699e5936ee3ea0d79e68d6c9b0d19eb0bf57`.
JUnit SHA256:`b6c4195a3b3136a971a255839d62083dd8ba1adc4a1942bbf4e56a7cdca15a20`.
The tested bootstrap's raw working-file SHA256 was
`0e0833664b478d90cc4249027f049c61ce877001cb6f13c025afa7d53fc35cc4`.
Git/platform packaging retains its own exact source/archive identities.

The previous22 redundant test chats were closed where still available and
archived, not deleted. The replacement leaf created no descendants and was
closed after its result. Main-only delegation and no recursive test dispatch
were made explicit in `AGENTS.md`.

No model, real graph/cache, training, inference, remote submission or protected
role was accessed by this verification. The failed remote profile remained
INFRASTRUCTURE_ONLY with no accepted timings. Its closed authority did not
release a retry, formal continuation or500K task.

## Profiler output durability follow-up

The shared observer exported Chrome traces directly to the final path. An
export exception could leave a partial `trace.json` and bypass the later error
summary. The repair exported to a sibling `.tmp` and replaced the final name
only after successful export. Failed exports removed their temporary file
where possible; cleanup faults were attached rather than replacing the first
exception. `profile.json` retained ERROR status, a missing trace digest and
the finalization error, while an existing caller exception stayed primary.
Instrumentation and the exclusive profiler lock were restored on either path.
Model arithmetic, schedules, update counts and scientific gates were unchanged.

The same Luna leaf ran `tests/test_family_execution_profile.py` directly.
An initial17-pass/2-failure run exposed an incorrect new test expectation that
the synthetic kernel originally had an `update_observer` attribute. Main fixed
the fixture assertion to require exact restoration of the original attribute
set; no production check was weakened. The subsequent full-file invocation
passed19 tests,0 failures/errors/skips,11.825 seconds, exit0. It included existing
state/RNG/callback equivalence, atomic publication and injected export/caller
faults. Both invocations used only bounded synthetic CPU fixtures and the same
host-only temp ACL plugin; no real cache/weights or remote device was opened.
The leaf created no descendants and was closed after its verdict.

Accepted artifacts:
`E:/MolGap/scratch/ppgt-bootstrap-fix-20261020/profile-durability-20261021/`.
Log SHA256:`6c0e5787d16edc993e918b2faaf56632d60602a3f57727bd845f704991a37dad`.
JUnit SHA256:`794a72c28be2c82d865ae3f531e9769f473fc269ea95065ae900bb398415d288`.
The earlier failed attempt remained under its original unique temp root.
This was CPU infrastructure verification, not remote profiling or a new RML
scientific trajectory. No allocation was retried or scientific result claimed.
