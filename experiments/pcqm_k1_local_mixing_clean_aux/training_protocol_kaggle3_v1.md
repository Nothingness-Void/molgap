# K1 V4 reference versus bounded joint aggregation, 2026-10-01

The user explicitly authorized a fresh original K1 V4 reference and one candidate,
then selected Kaggle3 (nvoid912). This supersedes the earlier no-reference-retraining
release constraint for this pair only. The historical qualification closures remain unchanged.

## Question and evidence

Can the preserved real-bond gated-message sum gain decision-relevant Gap accuracy
from a zero-initialized, degree1..4 bounded joint residual at layer6? Read
[evidence review](evidence_review.md) and the original [protocol](protocol.md)
for contextual evidence and expressivity limits. Historical P100 evidence is context;
the new same-job T4 reference owns the scientific comparison.
A gain could also arise from additional capacity; this pair cannot isolate that cause.

## Frozen execution

Two exclusive T4 workers, reference and ssma. Original 100K train and disjoint
50K internal development, seed42, physical batch128, FP32, no TF32 or EMA,
AdamW 4e-4/1e-5, clip1, cosine40/eta_min1e-6. Forty complete epochs,
31240 optimizer steps and 3998720 training presentations per arm.
Both strictly load the retained original tensor initialization and retain the
original Python epoch permutations. No chemical labels or pretraining.
All official validation and test roles remain sealed.

Before formal training both workers must qualify frozen source/config/data/state,
physical batch128, deterministic optimizer steps, checkpoint/resume serialization,
exact zero-added initial output and synchronized optimizer-inclusive candidate
overhead <=25%. Calibration exposure is recorded separately from formal exposure.
A qualification failure prevents both formal training workers from starting.

## Cost and decision

Authorize one T4x2 attempt, at most the Kaggle session limit (12 wall hours;
24 allocated T4 device hours upper bound). Expected cost remains estimated:
6 T4 device hours and 6 wall hours per arm plus bounded calibration; CPU/queue
unknown. These are planning allowances, not observed training costs.
Record each exclusive worker interval separately; bootstrap/queue and earlier
resume allocation remain explicitly missing. Do not claim complete total cost
from a lower-bound interval.

Candidate gate: same-job strict qualified reference, aligned finite 50K predictions,
>=3 meV improvement and paired row-bootstrap 95% interval entirely positive
(10000 draws, seed42). Row uncertainty is not training stochasticity.
Only independent accepted per-arm trace/exposure, selected/resume state, runtime,
roles, costs and V5/RML qualification can produce two complete replay entries.
No automatic 500K/full training, adoption or successor is authorized.

## Durability and terminal routing

Existing FamilyOutputSession owns atomic checkpoints, selected models, aligned
predictions and canonical trace. Existing experiment_cli owns plans/source/receipt;
Kaggle accelerator adapter owns submission. On return, bind actual job/version/source,
accept reference first and candidate second under the frozen same-run mapping.
Write attribution before another module. Adopted positive merges to desktop;
negative complete source/history routes to archive under BRANCHES.
