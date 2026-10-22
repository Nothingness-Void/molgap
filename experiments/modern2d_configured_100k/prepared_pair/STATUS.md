# Execution status

2026-10-22 22:35 JST: the exact version1 job completed both20-pass endpoints
within4h43m40s of measured Python allocation wall time. Independent output,
native saved-state and execution-retention checks passed. See the
[decision](results/decision.md) and [compact metrics](results/summary.json).
The existing Luna event was delivered and its heartbeat paused. No new compute
or automatic successor was submitted.

Canonical RML terminal staging failed closed on local attempt v2 versus physical
version1 labels; no old plan was rewritten and no Replay-Ready claim was made.
See [the exact closure blocker](results/rml_status.json).

## Launch observation

2026-10-22 17:51 JST: Kaggle3 confirmed RUNNING, kernel140182981/version1:
[exact job](https://www.kaggle.com/code/nvoid912/molgap-modern2d-prepared-pair-100k-v1).
One PPGT mean-readout and one native ESA arm, isolated T4 devices,20 planned
passes each, FP32/TF32off, BS128, fixed100K/CommonDev50K. Max9 wall/session
quota hours;18 physical GPU-hours. Native proof and scientific result remain
NOT_EVALUATED at launch; no automatic continuation or successor.

The private source dataset version1 was read back and all11 files matched
their SHA pins. Pulled kernel source/private/T4/mount/version checks passed;
the canonical launch receipt is ACCEPTED. Frozen source8c16b36e1,
Spec853f6e389e418330fc800e77d5ec7b68c063f3df469b02340ff57ef882dc5a70,
packageb48f6a5159744696238058fc38037bac4b7861d58c186407037437c6a9aaa14b.
Full operations and source bytes are external under
`E:/MolGap/artifacts/runs/modern2d-prepared-pair-100k-s42-v1/v2/`.
See [monitor binding](monitor.json) and [verification](verification.md).

Local attempt v1 stopped before prospective publication: its empty closed-route
context failed the RML input gate. Frozen v1 inputs were preserved; corrected
local attempt v2 retains the related budget-stop history. No GPU was submitted.
Two received HTTP errors during API transport repair were reconciled against
the exact absent target and complete authenticated owned listing before the
corrected SDK submission. Both original unknown records were retained; the
actual accepted kernel remained version1, with no invalid sources.
Scope and resource limits: [protocol](protocol.md), [authority](authority.json).
