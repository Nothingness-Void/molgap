# CPU release checks —2026-10-18

The actual two-arm declaration passed `check-workflow`: frozen family recipes
and prospective policy inputs were ready, without publishing or dispatching.
Draft-only checks first identified duplicated compatibility inventory entries
and incomplete supporting/closed-route planning facts. Those declaration
inputs were corrected before prospective publication or external run freezing;
no remote allocation was used. Spec3d2fe2faa7a7b0dd7136c319d19f19b5a70542c5356a11be26a23e69c3360b82
was the accepted prelaunch declaration.

The Luna test owner ran the new trial cases and existing workflow, output,
retrieval and logger regressions. An initial synthetic fixture error was
retained separately; its replacement used the actual frozen declaration.
The subsequent batch reported144 passed, one documentation-navigation failure
and one Windows symlink-permission skip. Adding the missing CLI link resolved
the navigation case; its six-case focused rerun passed. A second worker's
overlapping71-pass batch had the same navigation failure and skip and is not
counted as additional unique coverage. All14 protected user-file pins matched.

Independent source review identified process-group timeout cleanup and missing
trace-companion transport. The fixes retained the existing pair runtime,
session and retriever. The outer cutoff terminates/reaps the whole process
group; the pair stops early enough for cleanup. New session manifests bind
the actual companion SHA, selective retrieval includes it and the inspector
checks its hash/path/context. Historical manifests remain compatible.

The final changed-owner batch reported98 passed, one symlink-permission skip,
zero failures. It covered `tests/test_family_workflow_trial.py`, the navigation
case, `test_reform_retrieval.py`, `test_family_training_runner.py` and
`test_kaggle_family_retrieval.py`. The raw reports are under
`E:/MolGap/artifacts/reports/family-workflow-trial-v1-tests/`, especially
`trace-context-rerun.junit.xml`; earlier failed reports were not removed.
The implementation reviewer found no remaining P1 in the reviewed scope.

These CPU/source results authorize no scientific promotion and do not prove
the remote workflow. Actual bound submission,32-update endpoints, retrieval,
cost and storage observations remain separate mandatory acceptance evidence.
No coordinator test execution or new GPU POST occurred during these checks.
