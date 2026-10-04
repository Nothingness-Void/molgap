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
