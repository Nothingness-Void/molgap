# Execution preparation and scope — 2026-11-06 JST

Shared SOAP support was implemented independently on desktop f8f30b5b and
applied here as d8d68e18. The new experiment adds no model module or trainer.
It uses existing original K1 construction,recipe compiler,checkpoint/session,
native preflight,logging,submission journal,selective retrieval and RML owners.

Implementation provenance: author SOAP revision
`a1e553530fde97d0e6b307d7c82ac6d38b072340`,MIT license retained in k1_soap.py.
The upstream FP64 eigensolver retry is removed to honor frozen FP32;failure
propagates. First hidden-matrix call initializes GG/Q and skips weight update
and that group's WD;remainder AdamW still updates. Preserve31240kernel calls
and31239hidden updates rather than adding exposure. Same classifier as Muon.

The infrastructure owner executed affected existing CPU checks,not new default
nodes: fixed upstream12-call numerical equivalence through QR refresh,first-call
serialization/recovery,FP32 failure propagation,real K1 checkpoint/session/output
resume and mechanical inspection,and original Muon/PairToken regression. Three
affected checks passed in46.03s;high-level scientific compilation passed12.75s.
These are synthetic CPU qualification results,not T4 performance or model benefit.
They were not repeated by this task. Standard preparation rechecked actual inputs.

Kaggle3 official submit_workflow returned SUBMISSION_BOUND for kernel143051786/v1.
Frozen source d8d68e18be3d0b8ab0c1a2e2abde376da9da0ae4;Spec and package
identities are in STATUS.md and the retained launch receipt. Source,account,
quota,mount,explicit T4 request and actual returned version were bound by that
existing command. Luna owns subsequent scheduler observations.

Native preflight still must qualify each actual worker's repeat/resume,FP32,
resource use and output state. This file does not claim formal training results.
No adopted batched-Muon patch,SOAP-Muon hybrid,existing-job alteration,baseline
substitution,protected role,monitor service or successor was introduced.
