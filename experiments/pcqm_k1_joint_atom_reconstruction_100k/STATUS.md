# Operational status

2026-09-28: private Kaggle2 version 3 is confirmed submitted and QUEUED. It is
the sole active physical attempt. No training result or replay readiness exists.

## Active attempt

- Kernel: `kaseichou/molgap-k1-joint-atom-s42`, numeric `136108187`, version 3.
- Source: `bbc391d65136dd83558e755eaee1a433aa2d3605`; the private source dataset
  is ready. All six mounted source files were downloaded and hash-verified.
  Only attempt identity and local cancellation/recording adapters changed;
  the training recipe, encoder and target-transform bytes are unchanged.
- Both actual-reference-bound plans plus the separate `NO_TRAIN` audit are
  frozen in [attempts/v3](attempts/v3/source_config.json); RML validate,
  rebuild and frozen-check passed before release.
- The [v3 receipt](submission_receipt_v3.json) verifies actual version, ID,
  private visibility, source and data mounts, requested T4 and returned code.
  Device names and model preflight remain pending while the task is queued.
- The existing Luna B and original 30-minute heartbeat are ACTIVE through
  [monitor binding](monitor_binding.json). Healthy queue/run status is silent;
  terminal or actionable fault is handed to A once, without overriding A's
  model. The monitor checks actual latest version/ID because the installed SDK
  ignores the version suffix on its status endpoint.
  `CANCEL_ACKNOWLEDGED` now becomes an immediate idempotent `CANCELLED` event;
  the raw API status is retained and unacknowledged cancellation stays UNKNOWN.
- Retained output directory:
  `platforms/_records/kaggle/training/k1_joint_atom_s42_v3/`.
- Retry checks: 17 targeted static/synthetic tests passed. Source AST/inventory,
  actual-reference release gates and RML validate/rebuild/frozen-check passed.
  No local model execution or protected-role access occurred.

## Preserved cancelled attempt

Version 2 was independently confirmed `CANCEL_ACKNOWLEDGED`, with no exposed
outputs, logs or reason. Its two training plans and separate audit plan were
closed as `INFRASTRUCTURE_ONLY`; worker execution, role telemetry and native
cost were left unavailable. The controller event was finalized `NEXT_RUN_BOUND`
only after the actual v3 receipt. The user authorized one resubmission, not a
new scientific recipe. Authority: [cancellation decision](cancellation_v2/decision.md).
The original [v2 receipt](submission_receipt_v2.json), prospective files and
source package remain preserved. No v2 replay readiness is claimed.

## Preserved failed attempt

Version 1 reached ERROR before model preflight or training. Its failure and
three original plans were closed through the shared V5/RML failure path, with
unavailable trace/role telemetry left unavailable rather than fabricated.
Authority: [failure decision](failure_v1/decision.md).

- Kernel: `kaseichou/molgap-k1-joint-atom-s42`, numeric ID `136108187`.
- Source: `6576b4797b79308d1772a5fd6001ba86f0d74f43`, private source dataset
  `kaseichou/molgap-k1-joint-atom-source` v1, ready. All six published files were
  downloaded and independently matched to the frozen local package before push.
- Actual device names: two `Tesla T4`. Both workers stopped at the target
  normalization identity gate, before model construction or preflight.
- Returned code and all three data mounts match the release. Exact source and
  artifact digests are in [submission receipt](submission_receipt_v1.json).
- Its durable controller event was reconciled and finalized `NEXT_RUN_BOUND`.
  No new conversation or monitor was created for the repaired attempt.
- Two training trajectories and one separate fixed500K `NO_TRAIN` audit plan
  were frozen before submission. RML validate/rebuild/frozen-check passed.
- Focused implementation checks: 78 passed before submission; the post-submit
  raw `/code/` receipt normalization has three passing synthetic binding tests.
  No local training, model inference, or protected-role access occurred.

Raw publication, push, pulled code/metadata, startup snapshots and subsequent
outputs are retained under the ignored record directory
`platforms/_records/kaggle/training/k1_joint_atom_s42_v1/`. The Windows SDK source
upload failed before publication when given POSIX separators; retrying that
upload with a native absolute path succeeded. The one GPU push itself succeeded;
the SDK-returned `/code/` prefix was normalized for status lookup without a
second push or any change to the frozen training source.

The controller repair is recorded separately from the failed version; no
result, promotion, scientific successor or worker-initiated retry is asserted.
