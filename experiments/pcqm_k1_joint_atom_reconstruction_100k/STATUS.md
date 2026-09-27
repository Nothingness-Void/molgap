# Operational status

2026-09-27: private Kaggle2 version 1 reached ERROR before model preflight or
training. The controller reconciled the terminal run and is preparing one
infrastructure-only repair with the unchanged scientific recipe. No result or
replay readiness exists. Authority: [failure decision](failure_v1/decision.md).

- Kernel: `kaseichou/molgap-k1-joint-atom-s42`, numeric ID `136108187`.
- Source: `6576b4797b79308d1772a5fd6001ba86f0d74f43`, private source dataset
  `kaseichou/molgap-k1-joint-atom-source` v1, ready. All six published files were
  downloaded and independently matched to the frozen local package before push.
- Actual device names: two `Tesla T4`. Both workers stopped at the target
  normalization identity gate, before model construction or preflight.
- Returned code and all three data mounts match the release. Exact source and
  artifact digests are in [submission receipt](submission_receipt_v1.json).
- The existing Luna heartbeat is paused while the controller owns this failed
  attempt. A successful repaired receipt will rebind the same B and heartbeat;
  no new conversation or monitor will be created.
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
