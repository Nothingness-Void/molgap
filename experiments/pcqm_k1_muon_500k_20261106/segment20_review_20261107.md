# Verified first segment — 2026-11-07 JST

Muon retains a sizeable advantage through the first 20 of the contracted 60
passes. This supports continuing the existing question; it does not establish
the complete-schedule transfer gate. Both arms are PAUSED at the planned bound.

| Observation | AdamW | Muon |
|---|---:|---:|
| Completed passes | 20 | 20 |
| Optimizer steps | 78120 | 78120 |
| Sample presentations | 9999360 | 9999360 |
| Selected pass | 14 | 17 |
| Selected Live development MAE, eV | 0.115122742951 | 0.107192404568 |
| Pass20 Live development MAE, eV | 0.120407946408 | 0.108458697796 |
| Process wall, hours | 3.438401 | 7.551802 |
| Existing build-resume verification | RESUME_VERIFIED | RESUME_VERIFIED |

Selected Muon-minus-AdamW is -7.930338 meV; pass20 is -11.949249 meV.
All 20 matched pass endpoints favor Muon. The mean difference at passes16–20
is -9.142603 meV. Selected states differ in pass and are reported separately
from matched curve points. No paired interval was computed for this segment.
The results contradict absence of any optimizer benefit at this 500K prefix;
they do not exclude AdamW catching up during the remaining cosine schedule.
Cross-scale differences cannot isolate membership from changed cohort/exposure.

Muon process wall is 2.1963x AdamW. The physical allocation ledger is shared:
7.643057 entry-wall hours and 15.286114 allocated T4-device-hours, counted once.
Queue and time before Python entry remain unknown. The authority retains at most
three further physical sessions, 19.356943 entry-wall hours and 38.713886 allocated
T4-device-hours; these are budget balances, not runtime predictions. No changed
optimizer, precision, batching or schedule is authorized by this review.

Luna observed scheduler COMPLETE/latest1 matching receipt1 at 2026-11-06
16:22 UTC. There are no subsequent continuation receipts. The worker records
and resume verifier, rather than scheduler COMPLETE, establish the 20-pass bound.
The next preauthorized cumulative bound is step156240/pass40, followed by
step234360/pass60. This review itself submitted no continuation.

## Retained evidence

Pinned producer: job143011302/version1; original source, Spec and package remain
in STATUS.md and its launch receipt. Selective paused retrieval retained
135365489 bytes on D:, no full-output download. Root:
`D:/molgap-artifacts/retrieved/pcqm-k1-muon-500k-20261106-s42/v1/version1`.
Each arm retains canonical_trace.json, pause_receipt.json, selected model,
aligned development predictions, last checkpoint, runtime and cost evidence.
The existing verifier checked transported state without resetting training.

Verified resume manifests:
`D:/molgap-artifacts/resume-validation/pcqm-k1-muon-500k-20261106-s42/v1/20261107/{adamw,muon}/resume_bundle.json`.
AdamW SHA256: `33833bd335439a6046c9075cd4a3e824b262a2aca8fb2ae8aa50f8e8cf61eb88`.
Muon SHA256: `75a809177bd1bd24d834a28c893154a2db00539995ae348dae5bb27bec25ce42`.
Aligned curve CSV and raw command logs remain in the original run's
`v1/operations/segment20_curve_20261107.csv` and neighboring logs.

Strict terminal retrieval first returned BLOCKED at remote_manifests with only
RuntimeError and no message. The documented --paused route then succeeded;
paused retrieval has no terminal output manifest. The lost error detail was
reported to the infrastructure owner; no shared implementation was changed.
No terminal scientific decision or terminal RML record was fabricated for a
planned partial segment. Existing prospective trajectories remain open.
