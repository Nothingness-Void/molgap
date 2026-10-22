# Scoped verification before publication

On2026-10-22, the existing Luna worker ran bounded generated-data checks using
the project interpreter; no real-data model inference or training ran locally.

- Prepared recipe/certificate boundaries and real synthetic Session/resume:
  `test_modern2d_research_workflow.py test_modern2d_prepared_session.py`:
  19 passed.
- Existing formal model/Session, prepared forward and batched numerical checks:
  `test_modern2d_formal_training.py test_modern2d_prepared_execution.py
  test_modern2d_batched_execution.py`:109 passed.
- Saved native-gate rejection, SDK quota arithmetic/unknown status, physical
  vs session budget, monitor identity/idempotence, scientific-input and fake
  network submission journey:
  `test_kaggle_submission.py test_k1_relation_study_monitor.py
  test_modern2d_research_workflow.py test_modern2d_scientific_input.py`:
  52 passed in99.61s. These batches overlap; counts are not unique totals.

All used `--noconftest`, `PYTHONPATH=src`, process-only Git longpaths and short
scratch directories `C:/t/m2submit` or `C:/t/m2submit2`. One initial fixture setup
needed its missing scratch parent created. The worker was then closed.

The actual mixed-family declaration exposed a path-alias limitation: identical
accepted input manifests in the two family anchors were rejected by pathname.
The compiler was corrected to require exact accepted manifest SHA and matching
normalization/authorization semantics, retaining each family's feature checks.
The existing scientific-input suite passed; no new alias fixture was claimed.

Local v1 preparation stopped before prospective publication because its related
closed-route history was empty. Its original frozen input was retained. Local
v2 supplied the real budget-stop context and a separately bounded policy.

The actual v2 package passed native-free release-input checks, with both frozen
initial states, source closure, source/mount layout and prospective records;
local preparation took30.47s, including26.63s in the workflow owner.
Spec853f6e389e418330fc800e77d5ec7b68c063f3df469b02340ff57ef882dc5a70;
packageb48f6a5159744696238058fc38037bac4b7861d58c186407037437c6a9aaa14b;
source8c16b36e1ef446b75996fff052aa55197ad409e2;
archive5b41b2095f5ad6e94c31f5e89f8a8877326fe01f646fa9aaefc8813c7789a919.

This establishes local admission, not native parity, complete training, model
quality or Replay-Ready. The remote all-arm native gate remains mandatory.
Operation receipts live outside Git; only compact observations are integrated.
