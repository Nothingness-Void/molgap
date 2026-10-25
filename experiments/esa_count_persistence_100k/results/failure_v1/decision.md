# Prepared-proof consumer failure — 2026-10-25

Exact Kaggle2 kernel140716050/version1 ended ERROR. Both native preflight
workers returned0; both formal training processes returned1 before their first
optimizer update. The orchestrator's `training_started=true` records process
launch, not a completed training update. Native scratch updates are separate.

Both train logs stop in `modern2d_execution_proof.verify_repair_proof()` at
`Missing artifact: repeat-b.pt`. The preflight had atomically packed the
hash-bound repeat/resume states in `native_proofs.zip` and retired loose files.
Most consumers used `native_proof_storage.load_proof`; this execution-shadow
rechecker still required a loose file. Its accepted report and proof bytes were
not a missing model feature, CUDA OOM, or an unfavorable architecture result.

The repair uses the same lossless storage reader for both loose and archived
states, preserves exact allowed proof names and original digests, and retains
all resume, numerical-drift and native admission checks. Existing tests had
checked storage roundtrips separately and stubbed the execution-shadow reader
in a negative admission fixture; they did not cover its post-retirement read.
The existing actual CPU Session/resume journey was extended at that seam.

The Python allocation window was220.880103223s across two T4s, or
441.760206446 allocated device seconds. The inner220.472783932s window overlaps
and is not added. Queue, pre-Python provisioning and full billing remain unknown.
The settled account observation charged227.773 quota seconds after the original
zero-used observation; it was not represented as a per-job invoice.

Original logs, source identity, native certificates, execution reports and costs
were retained. [Failure observation](failure.json) binds the compact artifacts;
large proof archives remain available at the exact private version. No formal
prediction/endpoint or observed formal role audit exists. This is an
INFRASTRUCTURE_ONLY outcome with no scientific ranking or Replay-Ready claim;
canonical terminal finalization is not forced from absent formal evidence.

The human's standing unchanged-contract error-repair authorization supported
one bounded source retry after independent regression checks. Its remaining
wall ceiling was reduced to13,500s (3.75 session hours), rather than another
unaccounted4h allocation. The full observed account increment plus this ceiling
remained below the original4h envelope. New native qualification was required;
neither old certificates nor a positive CPU check released training by itself.
