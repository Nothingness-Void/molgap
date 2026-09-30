# CPU input recovery publication record

On 2026-09-30 the tested chunk-local indexing correction was packaged using
the existing `stage_preparation.py`, `stage_release_inputs` and Kaggle owning
adapter. The scientific input contract, fixed graph data, path policy and
degree initialization transform were unchanged. No GPU training was released.

Recovery version 1 attempted to mount the retained failed preparation kernel.
Kaggle returned kernel ID 136535008/version 1 but rejected that kernel source.
The independent API reconciliation confirmed ERROR before another publication.
Its raw response and uncertainty are retained in `submission_v1.json`; no
scientific result or measured CPU cost is inferred from the failed publication.
The original complete path chunks remain in ignored record storage.

Recovery version 2 removed the unavailable failed-kernel mount and rebuilt only
the deterministic path sidecar from the unchanged accepted fixed graph cache.
This avoided uploading approximately 1.94 GB of retained output. It still
required independent full source-row rederivation; construction alone could
not pass acceptance. The new contract, prospective plan and source package
were frozen separately rather than overwriting version 1.

`submission_v2.json` retains the actual returned kernel identity, version,
source/package digests and pulled-entry/metadata digests. At qualification the
API reported RUNNING, all invalid-source lists were empty, and the pulled
metadata had GPU disabled. Pulled code equaled staged code after CRLF/LF
transport normalization; raw digests were retained, not relabeled as identical
bytes. The private source dataset reported ready before publication.

Focused recovery/publisher tests passed (14 tests). `check-release` passed,
including selected clean imports, syntax, recipe SHA, exact frozen state and
upload/entry binding. The publishing adapter repeated that check before POST.
Its identity decoder was repaired to accept Kaggle's observed `/code/owner/slug`
reference format while still rejecting owner/URL conflicts. The wrapper also
returned nonzero for receipts requiring reconciliation, even if POST created a
kernel. A created job did not automatically mean valid inputs or runnable work.

The existing A/B binding and 30-minute heartbeat were reused; no new monitor
chat or automation was created. One qualified tick returned SILENT/RUNNING.
Terminal handoff preserves the controller's model/thinking settings and requires
independent saved-artifact CPU acceptance before separate G1/G2 GPU admission.

Prospective RML records exist for both recovery attempts. They are not
replay-ready training results or successful CPU terminal evidence. Acceptance,
observed roles/native cost and final RML closure must follow actual output.
The failed publication's prospective bytes remain preserved; its operational
terminal receipt is not a substitute for a finalized scientific trajectory.

The five-minute end-to-end target was not achieved. The delay included a
platform input-mount rejection, explicit reconciliation, source republication
and repeated release verification. No benefit, runtime qualification or model
result is claimed from these publication checks.
