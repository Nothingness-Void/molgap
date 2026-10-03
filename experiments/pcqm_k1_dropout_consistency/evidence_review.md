# Evidence review — 2026-10-04

Desktop review entry: docs/research/EDGE_STATE_RML_REVIEW_20261002.md; retained
owner report/catalog: D:/w/k1-flag/experiments/pcqm_k1_flag/analysis_report_zh.md
and rml_detailed_appendix.md. The catalog contains172 previous IDs and twoFLAG
records. Targeted RML/catalog search found no identical two-dropout-prediction
regression consistency constraint; unindexed archive coverage remains unknown.

Accepted FLAG terminal evidence: experiments/pcqm_k1_flag/kaggle3_v1/flag/
rml_finalized/v5_evidence.json on desktop; its terminal_decision.md and
attribution.md remain beside the owning experiment.
FLAG40epochs produced0.140606614589eV versus retained K1 0.141294460821eV;
gain0.687846meV,95% paired interval[-0.188802,1.636900]meV,below3meV gate.
Step ratio2.920258x; runtime/software differ from historical reference.
This supports retaining economical information paths and rejecting that specific
objective. It does not diagnose dropout instability,overfitting or why FLAG failed.

New hypothesis: imposing agreement across existing stochastic subnetworks may
improve clean regression beyond two-pass averaged supervision. Alternative:
agreement constrains useful variance and hurts; two-pass gradient averaging itself
explains gains; one-seed variation or historical runtime mismatch explains context.
This is an unproven falsifiable learning constraint,not a demonstrated model defect.

Closed SSMA,atom256,slot96,dense global attention,readout/JK,pretraining and router
questions remain closed. No repeated width,schedule,seed,FLAG step size or baseline
reference training is proposed. Primary same-run control isolates the explicit
consistency term; historical K1 is reused only with its original qualifications.
Cheapest implementation falsifier is synthetic mathematics plus bounded real
train-only remote qualification. Clean-development scientific gain requires the
complete frozen100K screen; no partial-curve early stop is installed.
