# Authorized continuation v2 of E2 kernel B (2026-11-13)

The user authorized (relayed 2026-11-13) one continuation of kernel `nothingnessvoid/molgap-gptrans-multihop-edge-e2-b-s48`
(run pcqm-gptrans-multihop-edge-20261112-e2-s48) that resumes mh_s48 only, from its v1 complete-epoch resume state to the frozen 60 passes.
ref_s48 finished all 60 passes in v1; its v1 terminal output is retained unchanged and pinned by its output-manifest sha256
(`retained_arms` in the recovery `workflow_report.json`). Gate, reference band and the 60-pass count are unchanged; no scientific,
recipe, init, data or source change. Wall ceiling 5400 s on account kaggle1. Earliest submission is after the 2026-11-14 00:00 UTC quota reset.

Infrastructure: partial-arm continuation, commit d73a0c20c291b6cc20f6f3ef65cce5215fe5e0cb, is the `recovery_repair` commit
(files `src/molgap/kaggle_pair_runtime.py` and `src/molgap/family_training_runner.py`).

Open precondition: the v1 output of this kernel is not yet retrieved locally, so the mh_s48 resume bundle and the ref_s48
manifest pin do not exist yet. Retrieve v1 first; no submission before both exist.
