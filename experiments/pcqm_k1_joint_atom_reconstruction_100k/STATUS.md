# Operational status

2026-09-27: private Kaggle2 version 2 is confirmed submitted and QUEUED. It is
the sole active physical attempt. No training result or replay readiness exists.

## Active attempt

- Kernel: `kaseichou/molgap-k1-joint-atom-s42`, numeric `136108187`, version 2.
- Source: `00bb679635eaf9fca208c6612f920afbc923d71b`; private source dataset v2
  is ready. All six mounted source files were downloaded and hash-verified.
- Both actual-reference-bound plans plus the separate `NO_TRAIN` audit are
  frozen in [attempts/v2](attempts/v2/source_config.json); RML validate,
  rebuild and frozen-check passed before release.
- The [v2 receipt](submission_receipt_v2.json) verifies actual version, ID,
  private visibility, source and data mounts, requested T4 and returned code.
  Device names and model preflight remain pending while the task is queued.
- The existing Luna B and original 30-minute heartbeat are ACTIVE through
  [monitor binding](monitor_binding.json). Healthy queue/run status is silent;
  terminal or actionable fault is handed to A once, without overriding A's
  model. The monitor checks actual latest version/ID because the installed SDK
  ignores the version suffix on its status endpoint.
- Retained output directory:
  `platforms/_records/kaggle/training/k1_joint_atom_s42_v2/`.
- Repair checks: 69 targeted tests passed; five additional monitor/receipt
  tests passed. No local model execution or protected-role access occurred.

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
