# Runtime recovery release — 2026-10-04

The first frozen GPU audit failed before workers under Kaggle Python 3.13.
Its [failure decision](results/failure_v1/decision.md) and immutable terminal
RML overlay preserve the original prospective bytes and measured allocation.

The recovery reused the platform accelerator adapter, source bundle owner,
RML planner/finalizer, atomic IO and existing A/B control store. New code is
limited to isolated interpreter setup, an import-only CPU wrapper and owning
failure/qualification evidence translation. No model architecture changed.

Kaggle2 CPU kernel `kaseichou/molgap-gptrans-runtime-qualification`, ID
136999674 version 1 was returned by the adapter and independently verified as
RUNNING. Pulled source matched the local entry after newline normalization;
the [receipt](environment_submission.json) and
[source/status observation](environment_remote_verification.json) own the
actual identity. It mounts only private audit inputs, not graph datasets.

The intended GPU v2 audit was planned but **not submitted**. Its frozen
prospective bytes were preserved through a pre-execution packaging correction
that made CPU-only and worker modes mutually exclusive. Both original and
corrected environment release bindings were retained. The corrected source
payload was checked remotely as an opaque `.bin`, avoiding automatic extraction.
Qualification tests the exact corrected entry and helper; passing imports does
not qualify GPU numerics. The original two-model reproduction barrier remains.

Luna's focused static/mocked suite reported 28 passed in 2.47 seconds. No local
model, inference, package install or GPU operation was used. Repository RML
validation passed after failure closure and both prospective plans. Existing
unrelated dirty derived files and user edits were preserved; no claim of a
new training Replay pair or synchronized derived corpus is made.

The same Luna B and existing 30-minute heartbeat were rebound to the exact CPU
job. Healthy status is silent; terminal events are sent once to A without
model/reasoning overrides. A owns acceptance, RML closure, unchanged-contract
GPU recovery, and subsequent scientific analysis before any training release.
