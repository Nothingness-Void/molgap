# Frozen diagnostic submission

On 2026-10-06 the private input dataset
`kaseichou/molgap-gptrans-bottleneck-inputs` returned `Ok`; its status was
independently observed as `ready` and all six EMA model files, three saved
prediction files, target transform, source archive and release metadata were
present in the remote file inventory. The first upload failed before dataset
creation because the Windows SDK resumable-upload path received forward
slashes; using a resolved native path succeeded. No kernel was duplicated.

The owning Kaggle adapter rechecked the frozen release report immediately
before submitting one T4 allocation. The returned identity was kernel
`kaseichou/molgap-gptrans-bottleneck-audit`, ID `137289711`, physical version
`1`, with no invalid source mount and no unresolved submission identity.
The actual receipt is [submission_receipt.json](submission_receipt.json).

The package was frozen at source commit
`13b21ee4b954c9de4a3bbafe9aad3aa07e750f0c`; complete source, EMA state,
prediction and transform bindings are in [release_binding.json](release_binding.json).
Prospective RML planning passed repository validation; rebuilding derived
outputs and `check --frozen` passed before submission. No terminal scientific
result, training Replay admission or causal amplitude conclusion was claimed.

The exact server-owned binding is [monitor_binding.json](monitor_binding.json).
The existing Luna B conversation owns silent healthy-state polling and one
idempotent terminal handoff to A. This release did not create another monitor,
authorize an optimizer step or release a successor.
