# V4 continuation release - 2026-10-06

Authority: the desktop user requested resubmission and continuation after the
[v3 inspection](../submission_v3/terminal_inspection/inspection_report.json).
Continue the same frozen Spec, two prospective trajectories, recipes, initial
lineage, pure-2D roles, FP32, seed42, optimizer and 60-epoch schedules.
K1 resumes 30 complete epochs/117180 steps; GPTrans resumes 45/175770.

Use a separate private checkpoint input
`nothingnessvoid/molgap-k1-gptrans-500k-resume-v3-s42`, with every artifact
required by each retained v3 stage manifest hash verified before publication.
Replace the prior resume mount rather than attaching both stages. The owning
preparer now accepts an explicit checkpoint dataset; runtime/trainer code is
unchanged. Reuse existing source packaging, release checks and Kaggle submitter.

Measured v2+v3 allocation is 22.468582 T4 hours. The next bounded9-hour/T4x2
invocation adds at most18 observed allocation hours, giving at most40.468582
against the48-hour ceiling before another continuation review. Version1
allocation and scheduler startup outside measured windows remain unknown.
GPTrans may finish this stage; K1 may require a later separately reviewed stage.
No protected evaluation, new question, automatic monitor or server handoff.
