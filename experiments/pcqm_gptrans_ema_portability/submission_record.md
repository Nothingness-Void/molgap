# Frozen EMA audit release — 2026-10-04

The user authorized audit first, result analysis second, then one justified
bounded training continuation. This release covered only NO_TRAIN inference.
The [frozen contract](contract.json) retained the earlier nomination criterion;
it did not release full training, extra seeds, or official evaluation roles.

Kaggle2 returned `kaseichou/molgap-gptrans-ema-portability-audit`, kernel ID
136990464, version1. The [receipt](submission_v1.json) had no invalid mounts or
identity conflict; the [initial scheduler observation](initial_status.json)
was RUNNING, not proof of completed inference or accepted metrics.

Source commit: `92a4b0ca7e0ee9f925c52a116e800af2e86f9cee`.
Exact source/input/contract and prospective bindings are in
[release_binding.json](release_binding.json); the prospective bytes were frozen
before any GPU submission. The earlier source commits recorded implementation
and an infrastructure-only opaque-archive correction, not GPU attempts.
The new registry could not encode NO_TRAIN honestly; a narrow release owner
reused existing source/IO, portable transform validation and the Kaggle adapter
instead of inventing a training recipe.

Before provisioning GPU, remote inventory showed Kaggle auto-expanded the
first `.tar.gz` source publication. Its original history was retained. Versions2
and3 published `source_payload.bin`; [remote verification](remote_input_verification.json)
checked actual inventory sizes and downloaded release bytes against the final
local release. Immutable data mirrors and their named development shards were
qualified from their accepted manifests; no graphs were rebuilt.

Luna's synthetic/static tests passed23 cases after correcting invocation-specific
reproduction barriers and setup/partial-launch cost retention. Packaged-source
CPU import probes passed without model construction, checkpoint loading or
inference. The owning adapter rechecked every staged source/model/payload byte,
entry script, metadata and target-transform asset immediately before the one POST.
These were mechanical release checks, not scientific acceptance.

The [existing monitor binding](monitor_binding.json) attached only this exact
server-owned job to the existing A/B loop. The prior30-minute heartbeat was
updated, not duplicated; healthy statuses were silent and terminal/fault delivery
was authorized to A without model/reasoning overrides. B could not release a
successor. The audit's accepted cost would count both allocated T4s, setup and
idle time; unobserved queue/teardown costs would stay unknown.

After both full original-role reproductions, paired inference could open the
fixed500K development role. Acceptance would recheck original saved payloads,
role order, chunk/source hashes, runtime isolation and budget before nomination.
NO_TRAIN evidence could be PAIRED_ENDPOINT, never STRICT_CAUSAL or a new training
Replay pair. A separately analyzed and released training contrast would require
its own prospective plan, comparison gate and actual Replay-ready closure.

## Interpreter recovery and audit v2 — 2026-10-04

The first GPU attempt stopped before workers because Kaggle Python3.13 could
not install the pinned torch2.4.1 wheel. Its evidence was closed separately as
[infrastructure-only](results/failure_v1/decision.md), without changing the
scientific contract. The [CPU qualification](results/environment_v1/decision.md)
accepted an isolated Python3.12.14 and the pinned imports; its NO_TRAIN RML
closure did not claim GPU numerical calibration or training Replay eligibility.

After reconciling the exact prior physical attempt, the owning adapter released
version2 of kernel136990464 on T4. The
[actual receipt](attempt_v2/submission_receipt.json) and
[remote verification](attempt_v2/remote_kernel_verification.json) identified
`kaseichou/molgap-gptrans-ema-portability-audit:v2`, initially RUNNING, and
verified the remote entry against the staged source. Frozen source, input,
contract and prospective identity remained in
[the v2 binding](attempt_v2/release_binding.json).

The existing B heartbeat was rebound, not duplicated. It was instructed to
retrieve version-pinned small metadata only, never recursively download the
isolated worker environment. A retained ownership of hash-bound tensor retrieval,
original-role reproduction, 500K-role acceptance and any subsequent training
decision. The original90-minute/3 allocated-T4-hour audit ceiling remained fixed.

The v2 terminal event subsequently reported ERROR. Both subprocesses stopped
at ambiguous cache-argument resolution before worker entry; the environment
qualified successfully. The [v2 failure decision](results/failure_v2/decision.md)
retained all six manifest-bound JSON files and measured allocation cost. Its
RML closure was INFRASTRUCTURE_ONLY, not a scientific negative. Two local
duplicate-mount regression cases passed before staging a new v3 source and
prospective binding; no frozen weight or scientific contract was modified.

## Dataset-scoped recovery v3 — 2026-10-04

The adapter released kernel136990464/version3 only after reconciliation of v2
ERROR, its infrastructure-only RML closure and local mount regression. The
[receipt](attempt_v3/submission_receipt.json) returned no mount/identity conflicts.
The [input verification](attempt_v3/remote_input_verification.json) accepted
private source dataset version6, exact inventory sizes and downloaded source
archive/manifest bytes. All older dataset versions were preserved.

The [new release](attempt_v3/release_binding.json) froze source commit
`1d78e653abf2657bc2536fe2ff5e46ccd3436073`, a new prospective attempt, and
the unchanged scientific contract and frozen model/payload hashes. Independent
[remote verification](attempt_v3/remote_kernel_verification.json) checked exact
physical version3, entry equality and RUNNING status. The existing B and
heartbeat were rebound to v3; the v2 event was finalized NEXT_RUN_BOUND.
No new training successor or scientific promotion was inferred from this release.
