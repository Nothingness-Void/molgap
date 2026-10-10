# Colab paired500K handoff

Owner: `codex/exp/k1-colab-500k-efficient` at
`D:/w/k1-colab-500k-efficient`. Do not launch from desktop integration or server.
The [protocol](protocol.md) owns the authorized single four-hour A100 window.

Notebook: https://colab.research.google.com/drive/11Ri_0aHTJ35kXBX1ml2JY84bL0Ix4eFq
Run ID: `k1-single-ema-500k-a100-20261010`, `attempt-001`.
The notebook ID is a locator, not an observed physical runtime identity.

Preparation observation on 2026-10-10:

- Source frozen at `c6f04e2d12c359f5b74e2615a3b638961858680f`.
- Prospective trajectory published through the existing RML planner; planned
  costs are estimates, not actual GPU usage.
- CPU acceptance verified all eleven fixed500K shards in private Drive.
- Thirty-nine targeted synthetic/adapter/BN tests passed. Frozen archive model
  construction and initialization loading passed without a forward.
- RML validate/rebuild/frozen check passed after restoring four hash-matched
  ignored historical checkpoints from desktop into this worktree. No historical
  evidence was rewritten; those copies are not training inputs for this question.
- One unrelated execution-profile test requires an ignored old payload checkpoint
  absent in this worktree. It was not regenerated or treated as a new-code failure.
- No A100 allocation or model execution had occurred at this observation.

Payload identity: [upload binding](submission/upload_binding.json).
Its local ignored ZIP is under `platforms/_records/colab/staging/` with the run ID.
Remote source payload destination:
`MyDrive/MolGap/V5/packages/k1_single_ema_500k_20261010.zip`.
Worker output destination:
`MyDrive/MolGap/V5/runs/k1-single-ema-500k-a100-20261010/attempt-001/worker`.

Before connecting A100, reconcile actual upload/attempt existence, record the
allocation start before connection, and install the prepared launch cell with
that original timestamp. Do not use Run All: CPU upload cells are not GPU work.
Do not reset the deadline or relaunch an existing attempt. The launch cell
requests runtime release on completion/failure and before the hard ceiling.
No periodic monitoring or automatic successor is authorized.

After return, inspect `terminal.json`, `last.json`, `preflight.json`, runtime,
allocation observation and artifact hashes. Only matched completed prefixes are
eligible for partial comparison. STOP_FOR_COST is not a full60 endpoint or
promotion. Populate actual role/cost/trace evidence through existing V5/RML
owners; missing facts stay missing. Route the branch only after terminal acceptance.
