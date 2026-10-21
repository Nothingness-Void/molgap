# Execution diagnostic bootstrap failure — 2026-10-21

Kaggle3 kernel139974457/version1
(`nvoid912/molgap-modern2d-execution-repair-v1`) ended ERROR.
The source and receipt bound commit329dca4f and package
e6735dde721be8546dc58176cbc2f1c4724a0a9e051c371781f7e34b0d752a0b.

The terminating exception was `ValueError: Unknown Modern2D startup purpose`
from the hash-pinned entrypoint's stdlib-only budget guard. The shared native
release owner understood CONFIGURED_EXECUTION_REPAIR but the distribution
bootstrap did not. This was an implementation/coverage omission, not a bad
model, dependency installation, optimizer update or scientific result.

The original guard ran before environment creation and native workers. No
train/Common-Dev graph payload, model, gradient, optimizer or profiler result
was reached. No endpoint, native equivalence or Replay readiness was claimed.
The failure's canonical RML finalization remains pending; retained prospective
records are not presented as closed trajectories.

[Retained evidence](retained/): actual launch receipt, source roundtrip
verification, platform response, startup traceback, and exact-version retrieved
bootstrap_allocation.json. The observed Python window was4.9443309609999915s,
two Tesla T4s,9.888661921999983 allocated device-seconds. Full billing, queue and
pre-Python provisioning remain unknown; this window must not be added to an
overlapping ledger or mistaken for complete billing.

Main repaired the missing bootstrap purpose branch without broadening its
1800s/1T4h/train-only/eight-update/two-family/reference-required scope. Luna's
33 focused bootstrap/release checks passed. A separately frozen
[controlled retry](../../retry_v2/protocol.md) was planned under the human
repair request. No continuation of model progress or automatic successor.
