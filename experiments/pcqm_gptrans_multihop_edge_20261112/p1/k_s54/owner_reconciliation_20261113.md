# Owner reconciliation: pcqm-gptrans-multihop-edge-20261112-s54 v1 (SUBMIT_UNKNOWN)

Date: 2026-11-13. Account nvoid912. Kernel nvoid912/molgap-gptrans-multihop-edge-p1-s54.

Facts: the journal recorded `SUBMIT_UNKNOWN` after the kernel push returned HTTP 404
(stage kernel_publication). The 404 came from Kaggle rejecting the default `requests`
User-Agent (fixed in commits c940c8b4, fe330264, 689be7b0), i.e. an HTTP-level rejection
before any kernel was created. The nvoid912 kernel list contains no s54 kernel, and the
submitter re-observes absence live before acting on this note.

Decision: the owner judges the kernel absent and the push failed before the POST was
accepted. This releases exactly one fresh POST for the same journal, slug, source dataset
and frozen package (no new attempt, no journal deletion, no slug change). Implemented by
`--owner-reconciliation/--owner-reconciliation-sha256` in `platforms/kaggle/submit_workflow.py`,
which moves SUBMIT_UNKNOWN -> SOURCE_PENDING (an existing PHASE_NEXT edge) and records this
note's path and SHA256 plus the released platform error in the journal. No science or frozen
gate is changed.
