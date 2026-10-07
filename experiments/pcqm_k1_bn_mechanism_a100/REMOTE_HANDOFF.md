# Colab binding

Desktop owns this frozen buffer-only diagnostic; no server takeover.

- Notebook: https://colab.research.google.com/drive/1QIO0Iy3PgCcmCoY1STIYW_KwenxwdDid
- Frozen package: upload_binding.json, SHA25607488e3a05eb7046b6a0eb35a2e91f4f8135e38af3455ae9eeb2913a4accdd7f.
- Durable output: MyDrive/MolGap/V5/runs/k1-bn-mechanism-a100-20261008/attempt-001.
- Result bundle: MyDrive/MolGap/k1-bn-mechanism-a100-results.zip.
- Source/prospective custody: 5dde43e4, 0b9573d8; exact executed bytes and
  serialized dependency origins are bound in payload_manifest.json and
  payload_import_check.json. The later prospective commit includes the buffer
  counter/invariant check present in the frozen payload.

Reconcile visible notebook/output before retry. Completion requires all four
case predictions and buffer invariants; accept locally with existing RML
finalize, then rebuild/check. Disconnect and delete the allocated runtime.
No training, model promotion, protected roles or automatic retry is released.
